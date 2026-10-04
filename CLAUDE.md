# CLAUDE.md

Read **[AGENTS.md](AGENTS.md)** first. It is the working agreement for this
repository — the hard lines, the leakage rules, the layout, and the list of
things that are still not true. It is the single source; this file exists only
because Claude Code loads `CLAUDE.md` by name and the agreement should not be
duplicated into two files that can drift apart.

Nothing else belongs here. If you want to add a rule, add it to `AGENTS.md`
and cite where the code enforces it; `tests/test_agents_md.py` checks that the
citation resolves.

The three things most often got wrong, in order:

1. **`.shift(1)`.** Every grouped rolling feature, no exceptions. If a feature
   can see game T while predicting game T, nothing else in the project matters.
2. **Make it fail first.** An assertion that passes before the fix is not a
   test. Revert the fix, watch the named test go red, put it back.
3. **Inspect, do not assume.** Read the actual file before assuming a column, a
   database field, an API payload key, a target name or a model path.
