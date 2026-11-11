"""The slips reach Discord, or say why not.

The defect this file guards: `build_dfs_entry_embed` existed, the payout engine
existed, the board existed — and the ONLY caller of the embed in the whole
repository was `scripts/nba_model_cli.py`. No scheduled run had ever built a
slip, so none could be filtered out. Nothing logged a refusal because nothing
tried.

Every guard below was mutation-tested: the fix reverted, the named test watched
go red, the fix put back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scheduler_worker as sw
from src.notify.discord import build_dfs_entry_embed
from src.quant.dfs_payouts import ProbabilitySource, load_payout_catalog
from src.quant.parlay_builder import SLIP_READY, build_slip
from src.utils.timezones import pacific_calendar_date


@pytest.fixture(scope="module")
def catalog():
    return load_payout_catalog()


def board_rows(n: int, *, event_prefix: str = "002260032") -> list[object]:
    """Rows shaped like `run_dispatch` builds them off the board CSV."""
    return [
        type("Row", (), {
            "slate_date": str(pacific_calendar_date()),
            "event_id": f"{event_prefix}{i}",
            "player_id": f"P{i}",
            "player_name": f"Player {i}",
            "target_market": "PTS",
            "side": "over",
            "line": 24.5,
            "model_prob": 0.57,
            "decision_status": "RECOMMENDED",
            "pickem_line": None,
        })()
        for i in range(n)
    ]


class Verdict:
    def __init__(self, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason


# --- the embed takes a slip at all ----------------------------------------


def test_the_entry_embed_prices_a_slip_and_not_only_a_routed_entry(catalog):
    """
    ONE EMBED FOR BOTH SHAPES. `build_dfs_entry_embed` read the WRAPPER's
    status, and `dfs_entry.EntryEvaluation` copies `payout.status` onto itself —
    so for that one type the two were always equal and the duplication was
    invisible. A slip wraps the same payout under its own vocabulary
    (SLIP_READY), and the wrapper check refused it although the entry WAS
    priced. The embed now reads the payout, which is the only object that knows.
    """
    slip = build_slip(board_rows(3), catalog["underdog:power:3"],
                      structure_key="underdog:power:3")
    assert slip.status == SLIP_READY

    embed = build_dfs_entry_embed(slip, publication=Verdict(True),
                                  slate_date="2026-11-10")
    assert "underdog:power:3" in embed["title"]
    assert "Not priced" not in json.dumps(embed), (
        "a priced slip was rendered as an abstention"
    )
    assert "3 legs" in embed["description"]
    assert len(embed["fields"]) == 3


def test_a_routed_entry_still_renders_after_the_status_read_moved(catalog):
    """
    THE REGRESSION THE CHANGE ABOVE COULD HAVE CAUSED. `dfs_entry` is the
    embed's original caller and must be unaffected.
    """
    from src.quant.contracts import MarketContext
    from src.quant.dfs_entry import PickemLeg, route_pickem_entry

    legs = [
        PickemLeg(
            leg_id=f"leg{i}",
            market=MarketContext(
                game_id=f"G{i}", market="PTS", player_name=f"Player {i}",
                line=24.5, payout_multiplier=3.0, is_pickem=True,
                status="VALID", source="operator",
            ),
            model_probability=0.57,
        )
        for i in range(2)
    ]
    entry = route_pickem_entry(catalog["underdog:power:2"], legs)
    embed = build_dfs_entry_embed(entry, publication=Verdict(True))
    assert "Not priced" not in json.dumps(embed), entry.reason


def test_every_leg_names_the_model_as_its_source_rather_than_a_question_mark(catalog):
    """
    A market-grounded leg and a model-grounded one are different claims and
    look identical once the column headers are gone. A leg printing "?" would
    read as provenance somebody forgot to record.
    """
    slip = build_slip(board_rows(2), catalog["underdog:power:2"], structure_key="k")
    embed = build_dfs_entry_embed(slip, publication=Verdict(True))
    rendered = json.dumps(embed)
    assert "MODEL" in rendered
    assert "?" not in " ".join(f["value"] for f in embed["fields"])
    assert slip.probability_source is ProbabilitySource.MODEL


def test_a_withheld_slate_replaces_the_slips_numbers_with_the_gates_reason(catalog):
    """
    THE GATE IS NOT BYPASSED BY A NEW EMBED. A model-sourced entry EV whose
    model has not been shown calibrated is exactly the figure that reads, in a
    channel, as though it had been.
    """
    slip = build_slip(board_rows(3), catalog["underdog:power:3"], structure_key="k")
    embed = build_dfs_entry_embed(
        slip, publication=Verdict(False, "nothing settled yet"),
    )
    rendered = json.dumps(embed)
    assert "nothing settled yet" in rendered
    assert "withheld" in rendered.lower()
    assert "EV" not in embed.get("description", "")


# --- the helper that turns a board into embeds ----------------------------


def test_the_gate_reaches_the_slips_through_the_helper_not_only_the_embed():
    """
    MUTATION THAT SURVIVED THE FIRST ATTEMPT: `_slip_embeds` passing
    `publication=None`, which publishes the numbers unguarded. The withheld
    test above calls `build_dfs_entry_embed` DIRECTLY, so it proved the embed
    obeys a verdict and said nothing about whether the helper hands one over —
    and the helper is the only path a scheduled run takes. This is the
    gate-bypass case, so it is tested where the bypass would live.
    """
    withheld = sw._slip_embeds(
        board_rows(9), slate_date=None,
        publication=Verdict(False, "nothing settled yet"),
    )
    assert len(withheld) == 4
    for embed in withheld:
        assert "nothing settled yet" in embed["description"]
        assert not embed["fields"], "a withheld slip still listed its legs"
        assert "%" not in embed["description"], (
            f"a withheld slip published a number: {embed['description']!r}"
        )
        assert "Player" not in embed["description"], (
            "a withheld slip named its legs; the gate strips every name from a "
            "withheld board and a new section must not reintroduce them"
        )

    allowed = sw._slip_embeds(
        board_rows(9), slate_date=None, publication=Verdict(True),
    )
    assert any("%" in e["description"] for e in allowed), (
        "the allowed case publishes nothing either, so the test above is vacuous"
    )


def test_one_embed_per_slip_size_reaches_the_channel():
    embeds = sw._slip_embeds(
        board_rows(9), slate_date="2026-11-10", publication=Verdict(True),
    )
    assert len(embeds) == 4
    titles = " ".join(e["title"] for e in embeds)
    for n in (2, 3, 4, 5):
        assert f"underdog:power:{n}" in titles


def test_a_board_too_short_for_a_slip_says_so_rather_than_sending_nothing():
    """
    AN EMPTY LIST IS INDISTINGUISHABLE FROM "NO SLATE" in a channel, which is
    how this pipeline reported success while emitting nothing. Four eligible
    legs must produce three priced slips and one embed naming the shortfall.
    """
    embeds = sw._slip_embeds(
        board_rows(4), slate_date=None, publication=Verdict(True),
    )
    assert len(embeds) == 4, [e["title"] for e in embeds]
    rendered = json.dumps(embeds)
    assert "4 eligible leg(s) on this board against the 5" in rendered, rendered[:400]
    assert sum(1 for e in embeds if "not built" in e["title"]) == 1, (
        "the one unbuildable size did not get its own named refusal"
    )


def test_a_same_game_board_produces_no_slip_and_names_the_reason():
    """
    THE USER'S CONSTRAINT, END TO END. Nine rows, all one game: nothing is
    buildable, and the channel is told which game blocked it rather than
    shown an empty section.
    """
    rows = board_rows(9, event_prefix="")
    for row in rows:
        type(row).event_id = "0022600321"
    embeds = sw._slip_embeds(rows, slate_date=None, publication=Verdict(True))
    rendered = json.dumps(embeds)
    # ONE embed, not four. All four sizes fail for the same reason, and four
    # near-identical refusals every night is how a channel learns to skip the
    # section. Every size is still named inside it.
    assert len(embeds) == 1, [e["title"] for e in embeds]
    for n in (2, 3, 4, 5):
        assert f"[{n}-leg]" in rendered, f"the {n}-pick refusal was dropped"
    assert "1 eligible leg(s)" in rendered, rendered[:400]


def test_muting_abstentions_drops_the_unpriced_slips_and_keeps_the_priced(monkeypatch):
    """`PROPIQ_DISPATCH_ABSTENTIONS=false` is the house switch for silence."""
    monkeypatch.setenv(sw.ENV_DISPATCH_ABSTENTIONS, "false")
    embeds = sw._slip_embeds(
        board_rows(4), slate_date=None, publication=Verdict(True),
    )
    assert len(embeds) == 3, "a shortfall embed survived the mute"
    assert all("Not priced" not in json.dumps(e) for e in embeds)


def test_an_unknown_platform_abstains_with_the_key_it_looked_for(monkeypatch):
    """
    Pricing a slip against whatever block happened to exist would be worse
    than saying nothing, and silently skipping reads as "that size is not
    offered".
    """
    monkeypatch.setenv(sw.ENV_SLIP_PLATFORM, "not_an_operator")
    embeds = sw._slip_embeds(
        board_rows(9), slate_date=None, publication=Verdict(True),
    )
    assert len(embeds) == 1
    rendered = json.dumps(embeds)
    assert "not_an_operator" in embeds[0]["title"]
    for n in (2, 3, 4, 5):
        assert f"not_an_operator:power:{n}" in rendered, (
            f"the {n}-pick lookup was not named"
        )


def test_the_flex_variant_is_selectable_and_has_no_two_pick(monkeypatch):
    monkeypatch.setenv(sw.ENV_SLIP_VARIANT, "flex")
    embeds = sw._slip_embeds(
        board_rows(9), slate_date=None, publication=Verdict(True),
    )
    rendered = json.dumps(embeds)
    assert "underdog:flex:2" in rendered, "the missing 2-pick flex vanished"
    assert "underdog:flex:5" in rendered


def test_an_unreadable_catalogue_names_the_refusal_rather_than_raising(monkeypatch):
    """
    A dispatch that raises here takes the board card down with it. The
    catalogue is a file on a volume and can be absent on a fresh container.
    """
    from src.quant.dfs_payouts import DfsPayoutError, load_payout_catalog

    def absent(*_a, **_kw):
        # The real message, raised the real way: `load_payout_catalog` refuses a
        # missing file rather than falling back to a table, because there is
        # deliberately no default table to fall back to.
        return load_payout_catalog(Path("config/does_not_exist.yaml"))

    monkeypatch.setattr("src.quant.dfs_payouts.load_payout_catalog", absent)
    with pytest.raises(DfsPayoutError):
        absent()

    embeds = sw._slip_embeds(
        board_rows(9), slate_date=None, publication=Verdict(True),
    )
    assert len(embeds) == 1
    assert "not found" in json.dumps(embeds)
    assert "no table is hardcoded" in json.dumps(embeds)


def test_the_slips_leave_room_for_the_board_card(monkeypatch, catalog):
    """
    Discord takes ten embeds per message and `send_embeds` refuses the WHOLE
    batch over that — so a long slip list would take the board card down with
    it. Two slots are reserved, for the board and the preview.

    MUTATION THAT SURVIVED THE FIRST ATTEMPT: deleting the trim. The first
    version of this test patched `parlay_builder.SLIP_LEG_COUNTS`, which
    `build_slips` binds as a DEFAULT ARGUMENT at definition time — so the patch
    did nothing, only four slips were ever produced, and four is under the
    limit. The assertion could not fail. `build_slips` is patched instead, so
    the trim is handed more than it may return and has to act.
    """
    from src.notify.discord import MAX_EMBEDS_PER_MESSAGE
    from src.quant.parlay_builder import build_slip

    rows = board_rows(9)
    one = build_slip(rows, catalog["underdog:power:2"], structure_key="underdog:power:2")
    assert one.status == SLIP_READY

    monkeypatch.setattr(
        "src.quant.parlay_builder.build_slips",
        lambda *_a, **_kw: [one] * (MAX_EMBEDS_PER_MESSAGE + 3),
    )
    embeds = sw._slip_embeds(rows, slate_date=None, publication=Verdict(True))
    assert len(embeds) == MAX_EMBEDS_PER_MESSAGE - 2, len(embeds)


# --- end to end through run_dispatch --------------------------------------


def write_board(path: Path, n: int, *, event_prefix: str = "002260032") -> None:
    import pandas as pd

    pd.DataFrame([
        {
            "slate_date": str(pacific_calendar_date()),
            "event_id": f"{event_prefix}{i}",
            "player_id": f"P{i}",
            "player_name": f"Player {i}",
            "target_market": "PTS",
            "side": "over",
            "line": 24.5,
            "model_prob": 0.57,
            "decision_status": "RECOMMENDED",
            "book_status": "DATA_NOT_AVAILABLE",
            "why": "model lean",
            "decision_basis": "model_lean",
        }
        for i in range(n)
    ]).to_csv(path, index=False)


@pytest.fixture
def dispatched(monkeypatch, tmp_path):
    """Run `run_dispatch` against a board on disk, capturing the embeds."""
    captured: dict[str, list] = {"embeds": []}

    def fake_send(embeds, **_kw):
        captured["embeds"] = [dict(e) for e in embeds]
        return type("R", (), {"status": "OK", "embeds_sent": len(embeds)})()

    monkeypatch.setattr("src.notify.discord.send_embeds", fake_send)
    monkeypatch.setenv(sw.ENV_DISPATCH, "true")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")

    def run(n: int = 9, **env):
        board = tmp_path / "board.csv"
        write_board(board, n)
        monkeypatch.setenv(sw.ENV_BOARD_CSV, str(board))
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        out = sw.run_dispatch()
        return out, captured["embeds"]

    return run


def test_a_scheduled_dispatch_now_carries_the_slips(dispatched):
    """
    THE WHOLE POINT. Before this, no scheduled run built a slip at all, so
    none could be filtered out — there was no writer. The board card still
    goes out; the slips are APPENDED, never substituted.
    """
    out, embeds = dispatched(9)
    assert out["status"] == "OK", out
    titles = [e.get("title", "") for e in embeds]
    assert any("Recommended plays" in t or "recommendation" in t.lower()
               for t in titles), f"the board card went missing: {titles}"
    slip_titles = [t for t in titles if "underdog:power:" in t]
    assert len(slip_titles) == 4, f"the slips did not reach dispatch: {titles}"


def test_the_slips_can_be_switched_off_without_touching_the_board(dispatched):
    out, embeds = dispatched(9, **{sw.ENV_DISPATCH_SLIPS: "false"})
    assert out["status"] == "OK", out
    titles = [e.get("title", "") for e in embeds]
    assert not [t for t in titles if "underdog:power:" in t]
    assert embeds, "switching the slips off silenced the board too"


def test_a_zero_prefixed_game_id_survives_the_board_csv(dispatched, monkeypatch):
    """
    PANDAS DROPS THE PREFIX. An NBA game id is `0022600321`; inferred as int64
    it becomes `22600321`. That is the defect the pre-lock check hit on this
    same file, and it matters more now that the slip builder groups by
    `event_id` to enforce one leg per game.
    """
    seen: list[object] = []
    real = sw._slip_embeds

    def capture(rows, **kw):
        seen.extend(rows)
        return real(rows, **kw)

    monkeypatch.setattr(sw, "_slip_embeds", capture)
    out, _embeds = dispatched(9)
    assert out["status"] == "OK", out
    assert seen, "dispatch built no rows"
    ids = {str(r.event_id) for r in seen}
    assert "0022600320" in ids, f"the leading zeros were stripped: {sorted(ids)}"
    assert all(len(i) == 10 for i in ids), sorted(ids)


# --- the card names people, not join keys --------------------------------


def test_a_slip_leg_is_labelled_with_the_player_not_the_join_key():
    """
    `SlipLeg.leg_id` keys on game, player, market, side and line --
    "0022601101:KCP:AST:over:0.5" -- and the embed printed it verbatim, so a
    card going out nightly named a game id where it should name a person.
    """
    from src.quant.parlay_builder import admit_leg

    leg = admit_leg({
        "event_id": "0022600001", "player_id": "espn:1", "line": 0.5,
        "player_name": "Andre Drummond", "target_market": "AST",
        "side": "over", "model_prob": 0.9, "decision_status": "RECOMMENDED",
    })
    assert leg.display_name == "Andre Drummond AST"
    assert "0022600001" not in leg.display_name
    assert ":" not in leg.display_name
    # The key is unchanged -- it is still what dedupes across slips.
    assert "0022600001" in leg.leg_id

    # AND THE EMBED MUST ACTUALLY USE IT. The first version of this test
    # asserted only the property, so reverting the embed to `leg_id` changed
    # nothing it could see and the mutation survived. The card is what goes
    # out, so the card is what is asserted.
    slip = build_slip(
        board_rows(2), load_payout_catalog()["underdog:power:2"],
        structure_key="underdog:power:2",
    )
    field_names = " ".join(
        f["name"] for f in build_dfs_entry_embed(slip, publication=Verdict(True))["fields"]
    )
    assert "Player 0" in field_names, field_names
    assert "002260032" not in field_names, (
        f"the card printed a game id where it should name a person: {field_names}"
    )


def test_a_leg_with_no_player_name_falls_back_to_the_id_not_a_blank():
    from src.quant.parlay_builder import admit_leg

    leg = admit_leg({
        "event_id": "0022600001", "player_id": "espn:1966", "line": 24.5,
        "player_name": None, "target_market": "PTS",
        "side": "over", "model_prob": 0.57, "decision_status": "RECOMMENDED",
    })
    assert leg.display_name == "espn:1966 PTS"


def test_the_routed_entry_path_keeps_its_own_leg_id(catalog):
    """
    `dfs_entry.LegResolution` has no `display_name`, and for that path the
    `leg_id` IS the caller's label. The fallback must not blank it.
    """
    import json

    from src.quant.contracts import MarketContext
    from src.quant.dfs_entry import PickemLeg, route_pickem_entry

    legs = [
        PickemLeg(
            leg_id=f"caller-label-{i}",
            market=MarketContext(
                game_id=f"G{i}", market="PTS", player_name=f"Player {i}",
                line=24.5, payout_multiplier=3.0, is_pickem=True,
                status="VALID", source="operator",
            ),
            model_probability=0.57,
        )
        for i in range(2)
    ]
    entry = route_pickem_entry(catalog["underdog:power:2"], legs)
    rendered = json.dumps(build_dfs_entry_embed(entry, publication=Verdict(True)))
    assert "caller-label-0" in rendered
