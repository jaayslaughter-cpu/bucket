# Fitting the fatigue multipliers — and why the constants did not change

Date: 2026-10-05. RESEARCH_ONLY. Module: `src/features/fatigue_fit.py`.
Tests: `tests/test_fatigue_fit.py`.

`features/fatigue_logic.py` folds a multiplier into **every** `{stat}_L2` this
pipeline publishes:

| flag | constant in force |
|---|---|
| `is_back_to_back` | **0.97** |
| `is_3_in_4` | **0.96** |
| `is_4_in_5` | **0.94** |
| away at DEN/UTA | **0.98** |

`docs/DATA_GAPS.md` lists them as unfitted heuristics. They were never measured,
and nothing downstream marks them uncertain, so a 3–6% haircut nobody checked
reaches every projection as though it had been checked.

## What the data says

Fitted on the real panel — **214,381 rows, ~212,000 usable** — as
`ratio(flagged) / ratio(unflagged)`, where each ratio is a volume-weighted
`sum(actual) / sum(baseline)` against `{stat}_L10`:

| flag | in force | PTS | REB | AST | n flagged |
|---|---|---|---|---|---|
| `b2b` | 0.970 | **1.0245** | **1.0233** | **1.0198** | 69,760 |
| `three_in_four` | 0.960 | **1.0252** | **1.0255** | **1.0296** | 73,641 |
| `four_in_five` | 0.940 | **1.0202** | **1.0283** | **1.0197** | 16,311 |

**Every fitted figure is above 1.0.** The constants in force are below it. On
three flags, across three stats, with tens of thousands of rows each, the naive
conditional comparison says fatigued players slightly *outperform* their
ten-game baseline — the opposite sign to a 3–6% haircut.

## So the constants were not changed, and will not be on this evidence

The obvious reading — "fatigue is a myth, set the multipliers to 1.02" — is
almost certainly wrong, and the reason is **survivorship**.

A player who appears on the second night of a back-to-back is a player healthy
enough to be dressed for it. The ones a coach would have rested, or whose knee
was sore, are *absent from the flagged rows entirely* — they are DNPs, or
scratches, and the panel has no row for them. So the flagged group is a
pre-selected fitter subset, and comparing it to everyone else measures
**selection, not fatigue**.

The figures above are therefore an honest measurement of the wrong quantity.
They are reported, and `fit.plausible` keeps them out of `multipliers` only
when they fall outside [0.85, 1.05] — these do not, so the module *would* offer
them. **Nothing applies them.** `fatigue_fit` never redefines a constant; a test
pins that.

## What would answer the question

This needs a design that holds the player fixed, not a different statistic:

1. **Within-player comparison.** Each player's flagged rows against their own
   unflagged rows, then aggregate the per-player effects. Removes the
   between-player composition that drives the result above.
2. **Condition on dressing.** Restrict both groups to players who played at all,
   and model the minutes separately — fatigue plausibly shows up as *fewer
   minutes*, not worse production per minute, and `{stat}_L10` carries the old
   minutes with it.
3. **Per-minute rate as the target.** `PTS / MIN` against the prior rate
   isolates efficiency from opportunity. If fatigue is a minutes story, the
   multiplier belongs on the minutes model, not on the stat.

(2) is the one I would do first: the current result is consistent with
"fatigued players play less but at the same rate", which the haircut models
badly and the minutes model would model well.

## Two fixes over the reference implementation

Adapted from `PropIQ_JuiceReel_Local`'s `fatigue_fit.py` — see
`docs/external_repo_review_2026-10.md` §1.2. Both change the answer, and each
has a test that fails without it.

**1. A ratio of totals, not a mean of ratios.** The reference computes
`mean(actual / baseline)`, clipped to [0.85, 1.05]. That statistic is pulled
upward hard by low-baseline rows — a bench player whose L10 is 2.0 points and
who scores 6 contributes a ratio of 3.0 — and the clip hides the bias rather
than removing it. Measured on the same panel:

| flag | ratio of totals | mean of ratios |
|---|---|---|
| `b2b` (PTS) | 1.0245 | **1.1176** |
| `b2b` (AST) | 1.0198 | **1.1690** |

The reference's statistic would have reported a **10–17% fatigue *boost***, and
the clip would have silently truncated it to 1.05.

**2. Measured against the unflagged rows, not against 1.0.** The reference takes
the flagged group's ratio as the multiplier, which assumes the unflagged ratio
is exactly 1.0. It is not — a `shift(1)` rolling baseline lags any trend, and
here `ratio_unflagged` runs 0.993–1.002. Small, but it is pure baseline bias
and folding it into a fatigue multiplier attributes it to fatigue.

## Status

The fitter ships; the constants stand. That is the honest state: we now know the
numbers in force are not supported by the naive comparison, and we know the
naive comparison is not the right test. Both facts are worth more than a
confident swap in either direction.
