"""Build a VERIFIED manifest of published official availability report URLs.

WHY A MANIFEST AT ALL. `collect_model_inputs` takes `--report-manifest`, a
JSON array of report URLs, to bypass fetching the season index. The
contribution that added it said the index "needs verification or a maintained
manifest" and shipped only an example file. This produces the real one.

WHY IT PROBES INSTEAD OF GENERATING. The filename pattern is regular --
`Injury-Report_YYYY-MM-DD_HH_MMAM.pdf` -- so a list could be printed from a
loop in four lines. That list would be a GUESS: it would contain URLs for days
with no games, for slots the league did not publish, and for dates before the
reports begin. Feeding guesses to a collector turns every one into a fetch
failure, and a manifest whose entries mostly 404 is worse than no manifest
because the failures bury the real ones. Every URL written here returned a
success to a HEAD request at the time of writing.

WHAT WAS MEASURED, 2026-11-13, so the numbers below are not assumed:

  * Reports persist. 2026-11-13 back to 2026-10-20 all answered 200.
  * 2026-10-18 and earlier answered 403, so 2026-10-20 is treated as the
    earliest available. A 403 is the CDN declining, and this script does not
    attempt to get around one.
  * Slots exist every 30 minutes across the whole day: every one of 12_00AM,
    05_30AM, 09_00AM, 12_30PM, 04_30PM, 05_00PM, 07_30PM, 08_00PM and
    11_30PM answered 200 on 2026-11-12.
  * A 200 is a real PDF, not an error page: a ranged GET returned bytes
    beginning `%PDF`.

WHAT THE SLOTS ARE FOR, and why several per day. `InputArchive.as_of` picks
the latest report at or before a cutoff, so the useful report for a game is
the last one published before its tip. Tips are spread across an evening, so
one report per day would be stale for the late games and absent for the early
ones. The default slots span the pregame window; denser is better evidence and
more requests, which is the caller's trade to make.

AND WHAT A BACKFILL CAN HONESTLY BE USED FOR -- the part worth reading twice.
`import-report` records `collected_at` as NOW, which is correct: we did not
hold these reports at the time. `as_of(mode="system")` requires publication
AND collection before the cutoff, so a backfilled report is correctly INVISIBLE
in system mode for a game that has already happened. It is visible in
`published` mode, which the archive labels "an explicitly counterfactual
historical replay".

That labelling is about BACKTESTS, where "did we actually have this?" is the
whole question, and in a backtest published mode would overstate what was
achievable. It is NOT a bar on TRAINING. A pregame report was published before
tip and was knowable to anyone who looked; learning the relationship between
pregame absences and outcomes from it is sound, because at serving time the
live collector supplies the same thing. So: train on published mode, serve on
system mode, and never quote a published-mode backtest as realised
performance.

RESEARCH_ONLY. This writes a list of URLs. It fetches no odds, prices nothing
and places no wager.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

#: The published host and path. Public, credential-free, and no query string --
#: `model_input_archive._url` refuses anything else.
BASE = "https://ak-static.cms.nba.com/referee/injury"

#: Earliest date that answered 200 on 2026-11-13. Earlier dates answered 403.
EARLIEST_AVAILABLE = date(2026, 10, 20)

#: Report slots to probe per day, spanning the pregame window so `as_of` has a
#: recent report for an early tip and for a late one.
DEFAULT_SLOTS = ("09_00AM", "01_30PM", "05_30PM", "07_30PM", "09_30PM")

#: Be a considerate client of a public CDN.
MAX_WORKERS = 6
TIMEOUT_SECONDS = 20


def report_url(day: date, slot: str) -> str:
    return f"{BASE}/Injury-Report_{day.isoformat()}_{slot}.pdf"


def _exists(session: requests.Session, url: str) -> tuple[str, int | None]:
    try:
        response = session.head(url, timeout=TIMEOUT_SECONDS, allow_redirects=True)
        return url, int(response.status_code)
    except requests.RequestException:
        return url, None


def build(
    start: date, end: date, slots: tuple[str, ...], *, verify: bool = True,
) -> tuple[list[str], dict[str, int]]:
    """Return the URLs that answered success, and a tally of what did not."""
    if start > end:
        raise SystemExit("start date is after end date")
    if start < EARLIEST_AVAILABLE:
        print(
            f"note: {start} precedes the earliest report that answered "
            f"({EARLIEST_AVAILABLE}); probing anyway and keeping what exists",
            file=sys.stderr,
        )

    candidates: list[str] = []
    day = start
    while day <= end:
        candidates.extend(report_url(day, slot) for slot in slots)
        day += timedelta(days=1)

    if not verify:
        return candidates, {"unverified": len(candidates)}

    tally: dict[str, int] = {}
    found: list[str] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "propiq-research/1.0"})
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for url, status in pool.map(lambda u: _exists(session, u), candidates):
                key = "unreachable" if status is None else str(status)
                tally[key] = tally.get(key, 0) + 1
                if status is not None and 200 <= status < 300:
                    found.append(url)
    return sorted(found), tally


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=EARLIEST_AVAILABLE.isoformat())
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument(
        "--slots", default=",".join(DEFAULT_SLOTS),
        help="comma-separated report slots, e.g. 05_30PM,07_30PM",
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="manifest path; pass to collect_model_inputs --report-manifest")
    parser.add_argument("--no-verify", action="store_true",
                        help="write the generated list WITHOUT probing (a guess, not a manifest)")
    args = parser.parse_args(argv)

    slots = tuple(s.strip() for s in args.slots.split(",") if s.strip())
    if not slots:
        raise SystemExit("no slots given")

    urls, tally = build(
        date.fromisoformat(args.start), date.fromisoformat(args.end), slots,
        verify=not args.no_verify,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(urls, indent=2) + "\n")
    print(f"wrote {len(urls)} verified URL(s) to {args.out}")
    print(f"responses: {dict(sorted(tally.items()))}")
    return 0 if urls else 1


if __name__ == "__main__":
    raise SystemExit(main())
