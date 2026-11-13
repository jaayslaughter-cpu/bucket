"""Append-only point-in-time inputs. RESEARCH_ONLY; never a serving feature.

Official reports are complete report snapshots, not per-player event streams.
PropLine IDs remain in the provider namespace until explicitly crosswalked.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from src.ingestion.id_crosswalk import normalise_player_name
from src.ingestion.propline import DFS_BOOKS, NBA_SPORT_KEY
from src.quant.contracts import plausible_american_odds
from src.utils.timezones import UTC

REPORT_PATTERN = re.compile(
    r"https://ak-static\.cms\.nba\.com/referee/injury/"
    r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf"
)
STATUSES = {"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "AVAILABLE"}
HEADER = ["GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"]


class InputError(ValueError):
    """Fail closed on missing provenance, schema drift, or ambiguous identities."""


def timestamp(value: str | datetime) -> datetime:
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise InputError("timestamp requires timezone")
        return result.astimezone(UTC)
    except (TypeError, AttributeError, ValueError) as exc:
        raise InputError("invalid timestamp or missing timezone") from exc


def report_time(url: str) -> datetime:
    match = REPORT_PATTERN.fullmatch(url)
    if not match:
        raise InputError("not an exact published official NBA report URL")
    try:
        local = datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %I %M %p")
    except ValueError as exc:
        raise InputError("invalid report timestamp") from exc
    return local.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)


def discover_report_urls(html: str, *, before: datetime) -> list[str]:
    class Links(HTMLParser):
        urls: set[str] = set()

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                url = dict(attrs).get("href", "")
                try:
                    if report_time(url) <= timestamp(before):
                        self.urls.add(url)
                except InputError:
                    pass

    parser = Links()
    parser.urls = set()
    parser.feed(html)
    return sorted(parser.urls, key=report_time)


def _text(words: list[dict]) -> str:
    """Join words by physical line, retaining multi-line reason text."""
    lines: dict[int, list[dict]] = defaultdict(list)
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        key = next((y for y in lines if abs(y - word["top"]) <= 2), round(word["top"]))
        lines[key].append(word)
    return " ".join(" ".join(w["text"] for w in sorted(lines[y], key=lambda w: w["x0"]))
                    for y in sorted(lines))


def report_player_name(name):
    if "," not in name:
        return name
    last, first = name.rsplit(",", 1)
    return f"{first.strip()} {last.replace(',', '').strip()}"


def parse_report_pages(pages: list[list[dict]]) -> list[dict]:
    """Parse seven-column word geometry; sparse game/team fields forward-fill.

    Status anchors are centered vertically against multi-line reason cells.
    Each word belongs to its nearest anchor, not its preceding text line.
    Missing/changing layouts fail the whole report rather than implying health.
    """
    boundaries = None
    context: dict[str, str] = {}
    rows = []
    for words in pages:
        headers = {w["text"]: w for w in words if w["text"] in HEADER}
        if "Matchup" in headers and "Reason" in headers:
            line = sorted([w for w in words if abs(w["top"] - headers["Matchup"]["top"]) <= 2],
                          key=lambda w: w["x0"])
            offset = 0
            for label in HEADER:
                start, combined = offset, ""
                while offset < len(line) and len(combined) < len(label):
                    combined += line[offset]["text"]
                    offset += 1
                if combined != label:
                    raise InputError("unrecognized split report header")
                headers[label] = line[start]
        if all(h in headers for h in HEADER):
            boundaries = [headers[h]["x0"] - 1 for h in HEADER]
            top = max(headers[h]["bottom"] for h in HEADER) + 1
        else:
            top = 55
        if boundaries is None:
            raise InputError("unrecognized report header")
        bottom = min((w["top"] for w in words if w["text"] == "Page"), default=555)
        cells = [[] for _ in HEADER]
        for word in words:
            if word["top"] < top or word["top"] >= bottom:
                continue
            column = next((i for i in range(6, -1, -1) if word["x0"] >= boundaries[i]), None)
            if column is not None:
                cells[column].append(word)
        anchors = sorted({round(w["top"], 1) for w in cells[5]})
        # A not-submitted team has no player/status; its reason is its anchor.
        anchors += [round(w["top"], 1) for w in cells[6] if w["text"].startswith("NOT")]
        anchors = sorted(set(anchors))
        if not anchors:
            raise InputError("report page has no status anchors")
        grouped = [[[] for _ in HEADER] for _ in anchors]
        for column, cell in enumerate(cells):
            for word in cell:
                row = min(range(len(anchors)), key=lambda i: abs(word["top"] - anchors[i]))
                grouped[row][column].append(word)
        for group in grouped:
            date, game_time, matchup, team, name, status, reason = map(_text, group)
            for key, value in [("game_date", date), ("game_time_et", game_time),
                               ("matchup", matchup), ("team", team)]:
                if value:
                    context[key] = value
            if not all(context.get(k) for k in ("game_date", "matchup", "team")):
                raise InputError("missing game/team context")
            try:
                game_date = datetime.strptime(context["game_date"], "%m/%d/%Y").date().isoformat()
            except ValueError as exc:
                raise InputError("invalid game date") from exc
            status = status.upper()
            if "NOT YET SUBMITTED" in reason:
                status, name = "NOT_YET_SUBMITTED", ""
            elif status not in STATUSES or not name:
                raise InputError("unknown status or missing player; layout may have changed")
            rows.append(context | {"game_date": game_date, "player_name": report_player_name(name) or None,
                                   "source_player_name": name or None,
                                   "status": status, "reason": reason})
    if not rows:
        raise InputError("empty report")
    return rows


def validate_report_title(title: str, expected_time: datetime):
    match = re.search(r"(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2})\s+(AM|PM)", title)
    if not match:
        raise InputError("missing internal report timestamp")
    local = datetime.strptime(" ".join(match.groups()), "%m/%d/%y %I:%M %p").replace(
        tzinfo=ZoneInfo("America/New_York"))
    if local.astimezone(UTC) != timestamp(expected_time):
        raise InputError("PDF internal timestamp differs from report URL")


def parse_report_pdf(content: bytes, *, expected_time=None) -> list[dict]:
    import pdfplumber

    if not content.startswith(b"%PDF"):
        raise InputError("response is not a PDF")
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = [p.extract_words(x_tolerance=1) for p in pdf.pages]
        if expected_time is not None:
            title = _text([w for w in pages[0] if w["top"] < 70]) if pages else ""
            validate_report_title(title, expected_time)
        return parse_report_pages(pages)


def _safe(value):
    """Unbounded traversal; never truncate before checking for credentials."""
    if isinstance(value, dict):
        for key, nested in value.items():
            compact = re.sub(r"[^a-z]", "", str(key).lower())
            if any(x in compact for x in ("apikey", "authorization", "password", "secret", "token")):
                raise InputError("credential-like key refused")
            _safe(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _safe(nested)


def _url(value):
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
        raise InputError("source URL must be credential-free HTTPS without query or fragment")


class InputArchive:
    """Content-addressed JSON and optional PDF blobs, atomically linked into place."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _put(self, folder: Path, content: bytes, suffix: str) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{hashlib.sha256(content).hexdigest()}{suffix}"
        fd, temporary = tempfile.mkstemp(dir=folder, prefix=".pending-")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(content)
                out.flush()
                os.fsync(out.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != content:
                    raise InputError("archive digest collision or corruption") from None
        finally:
            Path(temporary).unlink(missing_ok=True)
        return target

    def write(self, *, kind, source, source_url, observed_at, collected_at, status, rows, **metadata):
        if kind not in {"availability", "odds"} or status not in {"OK", "PARTIAL", "FAILED", "NOT_CONFIGURED"}:
            raise InputError("unknown snapshot kind/status")
        _url(source_url)
        observed, collected = timestamp(observed_at), timestamp(collected_at)
        if observed > collected:
            raise InputError("publication cannot be in the collection future")
        payload = dict(metadata, schema_version=1, kind=kind, source=source, source_url=source_url,
                       observed_at_utc=observed.isoformat(), collected_at_utc=collected.isoformat(),
                       status=status, rows=rows)
        _safe(payload)
        content = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        return self._put(self.root / "snapshots" / kind, content, ".json")

    def blob(self, pdf: bytes) -> str:
        if not pdf.startswith(b"%PDF"):
            raise InputError("only report PDF blobs may be archived")
        return str(self._put(self.root / "reports", pdf, ".pdf").relative_to(self.root))

    def as_of(self, kind, cutoff, *, mode="system", max_age_hours=6) -> list[dict]:
        """System mode requires publication AND actual local collection <= cutoff.

        Published mode is an explicitly counterfactual historical replay.
        Latest official FULL report wins globally, even when a team disappears.
        Odds select latest event snapshot then apply each pair's own time.
        """
        if mode not in {"system", "published"} or kind not in {"availability", "odds"}:
            raise InputError("unknown selection mode/kind")
        if not math.isfinite(max_age_hours) or max_age_hours <= 0:
            raise InputError("max age must be positive")
        cutoff = timestamp(cutoff)
        candidates: dict[str, list[dict]] = defaultdict(list)
        for path in sorted((self.root / "snapshots" / kind).glob("*.json")):
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != path.stem:
                raise InputError("archive checksum mismatch")
            snapshot = json.loads(content)
            published = timestamp(snapshot["observed_at_utc"])
            collected = timestamp(snapshot["collected_at_utc"])
            if published > cutoff or (mode == "system" and collected > cutoff):
                continue
            if cutoff - published > timedelta(hours=max_age_hours):
                continue
            key = "full_report" if kind == "availability" else snapshot.get("provider_event_id", "_collection")
            candidates[key].append(snapshot)
        result = []
        for values in candidates.values():
            # Failed attempts stay visible in system mode. In published replay,
            # never prefer a later parser result to the same report's failed attempt.
            latest = max((s["observed_at_utc"], s["collected_at_utc"]) for s in values)
            selected = [s for s in values if (s["observed_at_utc"], s["collected_at_utc"]) == latest]
            if len(selected) != 1:
                raise InputError("conflicting snapshots at identical timestamps")
            snapshot = dict(selected[0])
            if kind == "odds":
                snapshot["rows"] = [r for r in snapshot["rows"]
                                    if timestamp(r["observed_at_utc"]) <= cutoff
                                    and cutoff < timestamp(r["event_start_utc"])
                                    and cutoff - timestamp(r["observed_at_utc"])
                                    <= timedelta(hours=max_age_hours)]
                if (not snapshot["rows"] and snapshot["status"] in {"OK", "PARTIAL"}
                        and not snapshot.get("collection_summary")):
                    continue
            result.append(snapshot)
        return result


def normalize_odds(payload: dict, *, collected_at: datetime,
                   max_quote_age_minutes: float | None = 30) -> tuple[list[dict], list[dict]]:
    """Keep comparable two-sided PropLine quotes; do not invent opening prices.

    Supported schemas are the existing event/bookmaker/market/outcome contract.
    Unknown shapes and undated quotes are rejected, not guessed into training.
    """
    collected = timestamp(collected_at)
    if payload.get("sport_key") != NBA_SPORT_KEY:
        raise InputError("NBA sport_key required")
    if not payload.get("id"):
        raise InputError("missing provider event ID")
    event_start = timestamp(payload.get("commence_time"))
    rows, rejected = [], []
    for book in payload.get("bookmakers", []):
        source = str(book.get("key", "")).lower()
        if not source or source in DFS_BOOKS:
            rejected.append({"reason": "NON_COMPARABLE_BOOK", "bookmaker": source})
            continue
        for market in book.get("markets", []):
            key = market.get("key", "")
            if market.get("suspended_at"):
                rejected.append({"reason": "SUSPENDED_MARKET", "market": key, "bookmaker": source})
                continue
            grouped: dict[tuple, list[dict]] = defaultdict(list)
            for outcome in market.get("outcomes", []):
                if key == "spreads":
                    group = ("game", abs(float(outcome.get("point", math.nan))))
                elif key == "totals":
                    group = ("game", outcome.get("point"))
                elif key.startswith("player_"):
                    group = (outcome.get("description"), outcome.get("point"))
                else:
                    rejected.append({"reason": "UNSUPPORTED_MARKET", "market": key})
                    continue
                grouped[group].append(outcome)
            for (player, line), outcomes in grouped.items():
                try:
                    if not player or isinstance(line, bool) or not math.isfinite(float(line)):
                        raise InputError("INVALID_PLAYER_OR_LINE")
                    if len(outcomes) != 2:
                        raise InputError("MISSING_OR_DUPLICATE_SIDE")
                    names = [o.get("name") for o in outcomes]
                    expected = ({payload.get("home_team"), payload.get("away_team")} if key == "spreads"
                                else {"Over", "Under"})
                    if len(expected) != 2 or None in expected or set(names) != expected:
                        raise InputError("MISSING_OR_DUPLICATE_SIDE")
                    player_ids = {str(o["player_id"]) for o in outcomes if o.get("player_id")}
                    if len(player_ids) > 1:
                        raise InputError("DIFFERENT_PLAYER_IDS")
                    prices = []
                    times = []
                    for outcome in outcomes:
                        price = outcome.get("price")
                        if (isinstance(price, bool) or not isinstance(price, (float, int))
                                or not math.isfinite(price) or price != int(price)
                                or abs(price) < 100 or plausible_american_odds(price) is None):
                            raise InputError("INVALID_PRICE")
                        prices.append(int(price))
                        # The market snapshot proves the two sides coexisted;
                        # their last-change times need not be equal.
                        updated = timestamp(market.get("last_update"))
                        changed = timestamp(outcome.get("book_updated_at") or outcome.get("last_change_at")
                                            or updated)
                        if updated >= event_start:
                            raise InputError("POST_TIP_QUOTE")
                        if (changed > updated or updated > collected
                                or (max_quote_age_minutes is not None
                                    and collected - updated > timedelta(minutes=max_quote_age_minutes))):
                            raise InputError("FUTURE_OR_STALE_QUOTE")
                        last_seen = outcome.get("last_seen_at")
                        if last_seen and timestamp(last_seen) < updated:
                            raise InputError("WITHDRAWN_SIDE")
                        times.append(updated)
                    if len(set(times)) != 1:
                        raise InputError("DIFFERENT_SIDE_TIMESTAMPS")
                    pair = dict(zip(names, prices, strict=True))
                    row = {
                        "provider_event_id": str(payload["id"]), "provider": "propline",
                        "event_start_utc": event_start.isoformat(),
                        "bookmaker": source, "market": key, "line": float(line),
                        "player_name": None if player == "game" else player,
                        "provider_player_id": next(iter(player_ids), None),
                        "observed_at_utc": times[0].isoformat(),
                        "collected_at_utc": collected.isoformat(),
                        "side_metadata": [
                            {k: o[k] for k in ("name", "player_id", "book_updated_at",
                                              "last_change_at", "last_seen_at") if k in o}
                            for o in outcomes
                        ],
                    }
                    if key == "spreads":
                        home, away = payload["home_team"], payload["away_team"]
                        points = {o["name"]: float(o["point"]) for o in outcomes}
                        if points[home] != -points[away]:
                            raise InputError("ASYMMETRIC_SPREAD")
                        row.update(home_team=home, away_team=away, home_line=points[home],
                                   away_line=points[away], home_price=pair[home], away_price=pair[away])
                    else:
                        row.update(over_price=pair["Over"], under_price=pair["Under"])
                    rows.append(row)
                except (InputError, ValueError, TypeError, KeyError) as exc:
                    # No provider exception text or response bodies in logs/archive.
                    reason = str(exc) if isinstance(exc, InputError) else "INVALID_QUOTE_SCHEMA"
                    rejected.append({"reason": reason, "market": key, "bookmaker": source})
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[(row["bookmaker"], row["market"], row["player_name"], row["line"])].append(row)
    unique = []
    for group in by_pair.values():
        if len(group) == 1:
            unique.extend(group)
        else:
            rejected.append({"reason": "DUPLICATE_PAIR", "bookmaker": group[0]["bookmaker"],
                             "market": group[0]["market"]})
    return unique, rejected


def build_absence_export(snapshots, games, players):
    """Explicit deterministic crosswalk only. No invented IDs, zeros or usage.

    PLAYER_ID/GAME_ID follow the caller's panel namespace. This does NOT write
    the existing inactive cache, whose missing-coverage semantics differ.
    """
    game_map, player_map = defaultdict(set), defaultdict(set)
    for game in games:
        game_map[(game["game_date"], game["matchup"], game["team"])].add(
            (game["game_id"], game["team_abbreviation"]))
    for player in players:
        player_map[(player["team"], normalise_player_name(player["player_name"]))].add(player["player_id"])
    rows, unresolved = [], []
    for snapshot in snapshots:
        if snapshot["status"] != "OK":
            unresolved.append({"reason": "REPORT_NOT_OK", "status": snapshot["status"]})
            continue
        for row in snapshot["rows"]:
            matches = game_map[(row["game_date"], row["matchup"], row["team"])]
            identities = player_map[(row["team"], normalise_player_name(row["player_name"] or ""))]
            if row["status"] == "NOT_YET_SUBMITTED":
                unresolved.append(row | {"reason_code": "TEAM_NOT_SUBMITTED"})
            elif len(matches) != 1 or len(identities) != 1:
                unresolved.append(row | {"reason_code": "UNRESOLVED_OR_AMBIGUOUS_IDENTITY"})
            else:
                game_id, team = next(iter(matches))
                player_id = next(iter(identities))
                if not game_id or not player_id or not team:
                    raise InputError("blank crosswalk identity")
                rows.append({
                    "GAME_ID": game_id, "PLAYER_ID": player_id, "TEAM_ABBREVIATION": team,
                    "GAME_DATE": row["game_date"], "STATUS": row["status"],
                    "REASON": row["reason"], "observed_at_utc": snapshot["observed_at_utc"],
                })
    return rows, unresolved


def build_odds_export(snapshots, games, players):
    """Preserve provider IDs, and attach panel IDs only through explicit maps.

    Game mapping is provider_event_id -> game_id plus the participating teams.
    No provider ID is implicitly asserted to be an NBA or ESPN ID.
    """
    game_map, teams, player_map = defaultdict(set), defaultdict(set), defaultdict(set)
    for game in games:
        if game.get("provider_event_id") and game.get("game_id") and game.get("team"):
            event_id = str(game["provider_event_id"])
            game_map[event_id].add(game["game_id"])
            teams[(event_id, game["game_id"])].add(game["team"])
    for player in players:
        if player.get("team") and player.get("player_id") and player.get("player_name"):
            player_map[(player["team"], normalise_player_name(player["player_name"]))].add(player["player_id"])
    resolved, unresolved = [], []
    for snapshot in snapshots:
        for row in snapshot["rows"]:
            event_id = row["provider_event_id"]
            matches = game_map[event_id]
            if len(matches) != 1:
                unresolved.append(row | {"reason_code": "UNRESOLVED_OR_AMBIGUOUS_GAME"})
                continue
            game_id = next(iter(matches))
            player_id = None
            if row["player_name"] is not None:
                candidates = set()
                for team in teams[(event_id, game_id)]:
                    candidates.update(player_map[(team, normalise_player_name(row["player_name"]))])
                if len(candidates) != 1:
                    unresolved.append(row | {"reason_code": "UNRESOLVED_OR_AMBIGUOUS_PLAYER"})
                    continue
                player_id = next(iter(candidates))
            resolved.append(row | {"GAME_ID": game_id, "PLAYER_ID": player_id})
    return resolved, unresolved
