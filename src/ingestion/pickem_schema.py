"""
src/ingestion/pickem_schema.py — the capture contract for pick'em boards.

WHAT A PICK'EM ROW IS, AND IS NOT. Underdog, PrizePicks and Sleeper post a
LINE and sometimes a PAYOUT MULTIPLIER. They do not post two-way American
odds. EV and CLV need a two-way price, so a row from here is research
capture: `to_prop_line_row` forces ``is_pickem=True`` so `market_ev_gate`
abstains on it, and nothing in this module can produce a row that looks like
a de-viggable sportsbook price. `docs/pickem_props.md` is the standing rule.

WHY THE FIELD PATHS ARE CONFIGURATION AND NOT CODE. These are undocumented
internal endpoints. Their payloads get restructured without notice -- that is
not a hypothetical, it is the defect that took down the slate on 2026-10-18
when PropLine's own shape was assumed. So this module hardcodes NO provider
key names. A source declares, in `config/pickem_sources.yaml`, where its rows
live and which path each schema field reads, with CANDIDATES in order so a
recorded old shape and a new one can both be listed while a change is rolled
through. A path that resolves to nothing yields NO ROW and a named reason --
never a row with a guessed value in it.

THERE ARE NO DEFAULT PATHS IN THIS REPOSITORY, deliberately. Nobody here has
seen a live board response: this container cannot reach either host. Writing
plausible key names and calling them the provider's schema would be inventing
an API payload, which is the one thing the brief forbids outright. The
operator records one board response and fills the mapping; until then every
source reports NOT_CONFIGURED and names the file to edit.
`scripts/pull_pickem_props.py --describe` prints a recorded payload's real
paths so filling it in is reading, not guessing.

RESEARCH_ONLY. Captures posted lines with a timestamp. Prices nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

#: Snapshot states. `EMPTY` and `DATA_NOT_AVAILABLE` are NOT the same thing
#: and the whole contract rests on that: EMPTY means we read the board and it
#: posted nothing (the off-season, between slates); DATA_NOT_AVAILABLE means
#: we could not read it. Collapsing them is how a dark board reads as a quiet
#: night.
STATUS_VALID = "VALID"
STATUS_EMPTY = "EMPTY"
STATUS_UNAVAILABLE = "DATA_NOT_AVAILABLE"
STATUS_NOT_CONFIGURED = "NOT_CONFIGURED"

#: Out of scope by explicit instruction, whatever a config says. Checked as a
#: SUBSTRING of the board's own sport/league value, before the allowlist, so a
#: keep_values entry naming a college league cannot let one through.
FORBIDDEN_LEAGUES = ("ncaa", "college")

#: Schema fields a source mapping may fill. `player_name`, `market` and
#: `line` are required for a row to exist at all -- a line without a player,
#: a statistic or a number is not a partial row, it is not a row.
REQUIRED_FIELDS = ("player_name", "market", "line")
OPTIONAL_FIELDS = (
    #: BOTH SIDES' AMERICAN ODDS, where the board publishes them. Underdog
    #: does: the Apify actor returns `higher_american_odds` / 
    #: `lower_american_odds`, higher being OVER. Capturing a line without
    #: them and then flagging the row would discard a genuinely de-viggable
    #: two-way price, which is what the first version of this module did.
    "over_odds_american",
    "under_odds_american",
    #: PrizePicks' standard / goblin / demon. Read by `dfs_pricing`.
    "dfs_odds_type",
    #: ONE MULTIPLIER PER SIDE, because the board posts one per side. The
    #: Apify Underdog actor returns `higher_payout_multiplier` AND
    #: `lower_payout_multiplier`, and a boost on one side alone is exactly
    #: the case that makes a row look like an edge when it is a different
    #: product. Reading only one of them would miss half of those.
    "payout_multiplier",
    "payout_multiplier_under",
    "captured_at_utc",
    "team_abbr",
    "opponent_abbr",
    "game_time",
    "board_id",
    "player_id",
)
MAPPABLE_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS


class PickemConfigError(RuntimeError):
    """A source's mapping is missing or unusable. Never raised for an empty
    board: that is `STATUS_EMPTY`, which is data."""


@dataclass
class PickemPropLine:
    """One posted pick'em line, with the time it was observed."""

    source: str
    player_name: str
    market: str
    line: float
    #: The BOARD's own observation time where it reports one, else None.
    #: NEVER `now()`: a backfilled board that claims the moment we read it
    #: looks like a board that never moved. Same rule as
    #: `PropLineRow.captured_at_utc`.
    captured_at_utc: datetime | None = None
    #: As the board posted them. "Higher" is OVER and "lower" is UNDER.
    over_odds_american: int | None = None
    under_odds_american: int | None = None
    #: The board's own product label, where it has one (PrizePicks).
    dfs_odds_type: str | None = None
    #: The OVER/higher side's multiplier, and the UNDER/lower side's.
    payout_multiplier: float | None = None
    payout_multiplier_under: float | None = None
    team_abbr: str | None = None
    opponent_abbr: str | None = None
    game_time: datetime | None = None
    game_date: date | None = None
    board_id: str | None = None
    player_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def boosted_payout_multiplier(self) -> float | None:
        """
        Whichever side is furthest from 1.0, or None when neither is.

        A boost or discount on ONE side makes the whole row incomparable --
        the quoted prices no longer carry the full payout -- so the side that
        departs from 1.0 is the one `dfs_pricing` has to be asked about.
        Reading the over side alone would pass a row boosted on the under.
        """
        candidates = [
            value
            for value in (self.payout_multiplier, self.payout_multiplier_under)
            if value is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda value: abs(value - 1.0))

    def to_prop_line_row(self) -> Any:
        """
        The same row in the pipeline's own shape.

        WHETHER THIS IS A PICK'EM ROW IS NOT DECIDED HERE. The first version
        of this method forced ``is_pickem=True``, dropped both sides' odds and
        passed the multiplier straight through -- four decisions, every one of
        them contradicting the rule `ingestion/propline` had carried for
        months. `market_ev_gate` abstains when `is_pickem` is set OR
        `payout_multiplier` is not None, so for a standard Underdog pick at
        multiplier 1.0 -- which publishes real per-side American odds -- that
        threw away a de-viggable two-way price and then set two flags to make
        sure nothing could use it.

        `dfs_pricing.classify` is that rule, and both callers now read it.
        """
        from src.ingestion.dfs_pricing import classify
        from src.ingestion.propline import PropLineRow

        pricing = classify(
            self.source,
            dfs_odds_type=self.dfs_odds_type,
            payout_multiplier=self.boosted_payout_multiplier,
        )
        return PropLineRow(
            source=self.source,
            player_name=self.player_name,
            market=self.market,
            line=self.line,
            # As posted. A board that publishes no price leaves these None --
            # absent, never 0 and never a synthetic +100.
            over_odds_american=self.over_odds_american,
            under_odds_american=self.under_odds_american,
            nba_player_id=self.player_id,
            game_date=self.game_date,
            captured_at_utc=self.captured_at_utc,
            is_pickem=pricing.is_pickem,
            payout_multiplier=pricing.stored_payout_multiplier,
            status=STATUS_VALID,
            raw=self.raw,
        )


@dataclass
class PickemSnapshot:
    """One read of one board."""

    source: str
    status: str
    #: When WE read the board. Distinct from each line's own
    #: `captured_at_utc`, which is the board's claim; migration 003 split
    #: those two for the same reason.
    ingested_at_utc: datetime | None = None
    lines: list[PickemPropLine] = field(default_factory=list)
    message: str | None = None
    http_status: int | None = None
    #: Field paths that resolved to nothing, by schema field. An operator
    #: reading "no lines" needs to know whether the board was empty or the
    #: mapping is stale, and this is the difference.
    unresolved: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Read successfully, whether or not it held anything."""
        return self.status in (STATUS_VALID, STATUS_EMPTY)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "ingested_at_utc": (
                self.ingested_at_utc.isoformat() if self.ingested_at_utc else None
            ),
            "lines": len(self.lines),
            "message": self.message,
            "http_status": self.http_status,
            "unresolved": dict(self.unresolved),
        }


# ---------------------------------------------------------------------------
# the mapping: provider payload -> schema, declared in config
# ---------------------------------------------------------------------------


def resolve_path(payload: Any, path: str) -> Any:
    """
    Read one dotted path out of a payload, or return None.

    ``a.b.0.c`` walks dicts by key and lists by integer index. A missing key,
    an index past the end, or a scalar where a container was expected all
    return None rather than raising -- the caller turns that into a named
    unresolved field, which is reportable, where an exception would be a
    crashed capture.
    """
    if not path:
        return None
    current = payload
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError, TypeError):
                return None
            continue
        return None
    return current


def first_resolved(payload: Any, candidates: Any) -> tuple[Any, str | None]:
    """
    The first candidate path that resolves, with the path that won.

    CANDIDATES IN ORDER, not a single path, because an undocumented endpoint
    gets restructured and the recorded old shape has to stay listed while the
    change is rolled through. Returning WHICH path won is what makes a
    snapshot auditable afterwards.
    """
    if candidates is None:
        return None, None
    if isinstance(candidates, str):
        candidates = [candidates]
    for path in candidates:
        value = resolve_path(payload, str(path))
        if value is not None and value != "":
            return value, str(path)
    return None, None


def describe_payload(payload: Any, *, prefix: str = "", limit: int = 400) -> list[str]:
    """
    Every leaf path in a recorded payload, with a sample value.

    This exists so filling in a mapping is READING rather than guessing. The
    operator records one board response, runs ``--describe`` over it, and
    copies the paths it prints. No key name in this repository has to be
    anybody's recollection of what a provider sends.
    """
    out: list[str] = []

    def walk(node: Any, path: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            # Index 0 only: a board posts hundreds of structurally identical
            # rows and printing all of them buries the shape.
            if node:
                walk(node[0], f"{path}.0" if path else "0")
            else:
                out.append(f"{path}  (empty list)")
        else:
            sample = repr(node)
            out.append(f"{path}  = {sample[:60]}")

    walk(payload, prefix)
    return out


#: Where the mapping lives. Overridable so a test needs no fixture in the
#: repository root and a deployment can mount its own.
DEFAULT_CONFIG_PATH = "config/pickem_sources.yaml"
ENV_CONFIG_PATH = "PROPIQ_PICKEM_CONFIG"


@dataclass(frozen=True)
class PickemSourceConfig:
    """One source's declared wiring, as read from the mapping file."""

    name: str
    enabled: bool
    board_url: str | None
    rows_paths: tuple[str, ...]
    fields: dict[str, tuple[str, ...]]
    lookups: dict[str, dict[str, Any]]
    sport_filter_paths: tuple[str, ...]
    sport_keep_values: tuple[str, ...]
    local_dir: str | None
    market_aliases: dict[str, str]
    #: How the bytes arrive: "http" (a public board URL), "local" (operator-
    #: supplied JSON) or "apify" (a paid Actor run).
    transport: str = "http"
    #: The `apify` block, when transport is "apify".
    apify: dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        """
        Whether this source can produce a row at all.

        `enabled` IS PART OF THIS, and that matters more since a transport
        can cost money: the Apify actors bill per line. A flag that reads as
        off and is never consulted is how an unattended scheduled capture
        spends somebody's balance. A board URL with no row path, or row paths
        with no required field, is likewise not configured -- reporting
        configured and then returning nothing is the "quiet night" failure
        this contract exists to prevent.
        """
        if not self.enabled:
            return False
        if not self.rows_paths:
            return False
        return all(self.fields.get(name) for name in REQUIRED_FIELDS)

    def why_not_configured(self) -> str:
        """What exactly to fill in, by name."""
        missing: list[str] = []
        if not self.enabled:
            missing.append(
                "`enabled: true` (off by default — a capture through a paid "
                "transport must be switched on deliberately)"
            )
        if not self.rows_paths:
            missing.append("`rows` (the path to the array of lines)")
        for name in REQUIRED_FIELDS:
            if not self.fields.get(name):
                missing.append(f"`fields.{name}`")
        config = os.environ.get(ENV_CONFIG_PATH) or DEFAULT_CONFIG_PATH
        return (
            f"{self.name} is not configured: {', '.join(missing)} is empty in "
            f"{config}. Record one board response and run "
            f"`python -m scripts.pull_pickem_props --describe <file>` to see "
            f"its real paths — this repository ships no default paths because "
            f"nobody here has seen a response, and guessing them would invent "
            f"an API payload."
        )


def _as_paths(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value if str(v).strip())


def load_source_config(
    name: str, path: str | None = None
) -> PickemSourceConfig:
    """Read one source's wiring. Raises only when the FILE is unusable."""
    from pathlib import Path

    import yaml

    resolved = Path(path or os.environ.get(ENV_CONFIG_PATH) or DEFAULT_CONFIG_PATH)
    if not resolved.is_file():
        raise PickemConfigError(
            f"no pick'em source mapping at {resolved}. There is no built-in "
            f"default: see docs/pickem_props.md."
        )
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    sources = payload.get("sources") or {}
    if name not in sources:
        raise PickemConfigError(
            f"{resolved} declares no source {name!r}; it has "
            f"{sorted(sources) or 'none'}."
        )
    spec = sources[name] or {}

    # Board spelling -> canonical stat, inverted once here so the parser does
    # one dict lookup per row instead of scanning the alias lists.
    aliases: dict[str, str] = {}
    for stat, spellings in (payload.get("market_aliases") or {}).items():
        for spelling in spellings or []:
            aliases[str(spelling).strip().lower()] = str(stat)
        aliases[str(stat).strip().lower()] = str(stat)

    url_env = spec.get("url_env")
    board_url = (os.environ.get(url_env) if url_env else None) or spec.get("board_url")
    sport = spec.get("sport_filter") or {}

    return PickemSourceConfig(
        name=name,
        enabled=bool(spec.get("enabled")),
        board_url=(str(board_url).strip() or None) if board_url else None,
        rows_paths=_as_paths(spec.get("rows")),
        fields={
            key: _as_paths((spec.get("fields") or {}).get(key))
            for key in MAPPABLE_FIELDS
        },
        lookups=dict(spec.get("lookups") or {}),
        sport_filter_paths=_as_paths(sport.get("field_paths")),
        sport_keep_values=tuple(
            str(v).strip().lower() for v in (sport.get("keep_values") or [])
        ),
        local_dir=spec.get("local_dir"),
        market_aliases=aliases,
        transport=str(spec.get("transport") or "http").strip().lower(),
        apify=dict(spec.get("apify") or {}),
    )


LOOKUP_PREFIX = "lookup:"


def _build_lookups(
    payload: Any, declared: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Index each declared side-table by its id, once per payload.

    A normalised board puts players, games and lines in sibling arrays keyed
    by id. Without this the mapper could read only what is nested inside a
    row, which for such a payload is ids and nothing a human recognises.
    """
    built: dict[str, dict[str, Any]] = {}
    for name, spec in (declared or {}).items():
        array_path = str((spec or {}).get("path") or "")
        id_field = str((spec or {}).get("id") or "id")
        entities = resolve_path(payload, array_path)
        index: dict[str, Any] = {}
        if isinstance(entities, list):
            for entity in entities:
                key = resolve_path(entity, id_field)
                if key is not None:
                    index[str(key)] = entity
        built[str(name)] = index
    return built


def _resolve_field(
    row: Any,
    candidates: tuple[str, ...],
    lookups: dict[str, dict[str, Any]],
) -> tuple[Any, str | None]:
    """One schema field, honouring `lookup:` indirection."""
    for path in candidates:
        if path.startswith(LOOKUP_PREFIX):
            try:
                _, table, id_path, inner = path.split(":", 3)
            except ValueError:
                continue
            key = resolve_path(row, id_path)
            if key is None:
                continue
            entity = (lookups.get(table) or {}).get(str(key))
            value = resolve_path(entity, inner)
        else:
            value = resolve_path(row, path)
        if value is not None and value != "":
            return value, path
    return None, None


def _as_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    import math

    return number if math.isfinite(number) else None


def _as_datetime(value: Any) -> datetime | None:
    """A board timestamp, or None. Never now(): see `captured_at_utc`."""
    if value is None:
        return None
    from src.ingestion.propline import _parse_epoch, _parse_iso

    return _parse_iso(value) or _parse_epoch(value)


def rows_from_payload(
    payload: Any, config: PickemSourceConfig
) -> tuple[list[PickemPropLine], dict[str, int], dict[str, int]]:
    """
    Turn one board response into rows, under the source's declared mapping.

    Returns ``(lines, unresolved, dropped)``. Nothing here raises on a shape
    it does not recognise and nothing here fills a field it could not read:
    a row missing a player, a statistic or a number is DROPPED and counted,
    because a pick'em line with a guessed value in it is indistinguishable
    downstream from a real one.
    """
    lookups = _build_lookups(payload, config.lookups)
    rows, _ = first_resolved(payload, list(config.rows_paths))
    if not isinstance(rows, list):
        return [], {"rows": 1}, {}

    lines: list[PickemPropLine] = []
    unresolved: dict[str, int] = {}
    dropped: dict[str, int] = {}

    def miss(field_name: str) -> None:
        unresolved[field_name] = unresolved.get(field_name, 0) + 1

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for row in rows:
        if config.sport_filter_paths:
            sport, _ = _resolve_field(row, config.sport_filter_paths, lookups)
            if sport is None:
                miss("sport")
                drop("sport not readable")
                continue
            # NCAA FIRST, AND IT OVERRIDES THE OPERATOR'S OWN ALLOWLIST.
            # Order matters and it was wrong: this sat AFTER the keep_values
            # check, where it was unreachable dead code -- "ncaa" cannot be a
            # substring of "nba", so an exact-match allowlist had already
            # excluded college by construction. A mutation that deleted the
            # check entirely passed every test. Decoration that reads like a
            # safeguard is worse than none, because the next person widens
            # keep_values believing it still guards them. Here it is the
            # guard: college is out of scope by explicit instruction, so no
            # config can opt back into it.
            if any(bad in str(sport).strip().lower() for bad in FORBIDDEN_LEAGUES):
                drop("NCAA/college, out of scope")
                continue
            if str(sport).strip().lower() not in config.sport_keep_values:
                drop("not NBA")
                continue

        player, _ = _resolve_field(row, config.fields.get("player_name", ()), lookups)
        if player is None or not str(player).strip():
            miss("player_name")
            drop("no player")
            continue

        # MARKET RESOLVES DIFFERENTLY FROM EVERY OTHER FIELD, and it has to.
        # `first_resolved` takes the first candidate path that holds a value;
        # for a market the first path that holds a RECOGNISED STAT is what is
        # wanted, and those are not the same. The Apify Actor emits both
        # `stat` (machine key) and `stat_display` (display name), and its
        # listing publishes NBA display names but not the machine keys for
        # the compound stats -- so an unpublished key in `stat` would win the
        # race and drop a row that `stat_display` maps perfectly well.
        market, raw_market = None, None
        for path in config.fields.get("market", ()):
            value, _ = _resolve_field(row, (path,), lookups)
            if value is None:
                continue
            if raw_market is None:
                raw_market = value  # remembered for the message
            hit = config.market_aliases.get(str(value).strip().lower())
            if hit is not None:
                market, raw_market = hit, value
                break
        if raw_market is None:
            miss("market")
            drop("no market")
            continue
        if market is None:
            drop(f"unmapped market {str(raw_market)[:40]!r}")
            continue

        raw_line, _ = _resolve_field(row, config.fields.get("line", ()), lookups)
        line = _as_float(raw_line)
        if line is None:
            miss("line")
            drop("no numeric line")
            continue

        multiplier = _as_float(
            _resolve_field(row, config.fields.get("payout_multiplier", ()), lookups)[0]
        )
        multiplier_under = _as_float(
            _resolve_field(
                row, config.fields.get("payout_multiplier_under", ()), lookups
            )[0]
        )
        # Parsed by PropLine's own reader: "-120" is a price, "1.84" is
        # DECIMAL odds and must not be mistaken for one, and 0 is not a price.
        from src.ingestion.propline import _american

        over_odds = _american(
            _resolve_field(row, config.fields.get("over_odds_american", ()), lookups)[0]
        )
        under_odds = _american(
            _resolve_field(row, config.fields.get("under_odds_american", ()), lookups)[0]
        )
        captured = _as_datetime(
            _resolve_field(row, config.fields.get("captured_at_utc", ()), lookups)[0]
        )
        game_time = _as_datetime(
            _resolve_field(row, config.fields.get("game_time", ()), lookups)[0]
        )

        def text(field_name: str, _row: Any = row) -> str | None:
            # `_row` is bound at definition rather than closed over: the
            # closure form is correct today only because every call happens
            # inside this iteration, and that is one refactor away from
            # reading the last row's value for every row.
            value, _ = _resolve_field(_row, config.fields.get(field_name, ()), lookups)
            return str(value).strip() or None if value is not None else None

        game_date = None
        if game_time is not None:
            from src.utils.timezones import pacific_calendar_date

            # Pacific CALENDAR day, as everywhere else in this project: a 7pm
            # PT tip belongs to that day's slate, not to the UTC date.
            game_date = pacific_calendar_date(game_time)

        lines.append(
            PickemPropLine(
                source=config.name,
                player_name=str(player).strip(),
                market=market,
                line=line,
                captured_at_utc=captured,
                over_odds_american=over_odds,
                under_odds_american=under_odds,
                dfs_odds_type=text("dfs_odds_type"),
                payout_multiplier=multiplier,
                payout_multiplier_under=multiplier_under,
                team_abbr=text("team_abbr"),
                opponent_abbr=text("opponent_abbr"),
                game_time=game_time,
                game_date=game_date,
                board_id=text("board_id"),
                player_id=text("player_id"),
                raw=row if isinstance(row, dict) else {"row": row},
            )
        )

    return lines, unresolved, dropped
