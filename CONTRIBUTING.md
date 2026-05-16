# Contributing

Thank you for taking part in Learn2Design-2026. This document covers two
audiences: participants who want to submit an algorithm, and external
contributors who want to improve the starting kit, docs, or baselines.

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## Submitting an algorithm (participants)

1. Fork this repository.
2. Place exactly one `.py` file containing your `OptimizationAlgorithm`
   subclass in `submit/<your-handle>/`. If you need extra packages, add a
   `requirements.txt` next to it. If you need bundled weights or data, place
   them in the same directory.
3. Add a `metadata.yaml` in the same directory with:
   ```yaml
   handle: your-github-handle
   team: [optional list of team members]
   algorithm_name: my_algo
   description: one-line summary
   ```
4. Open a pull request with the title `[submission] <your-handle>: <algo name>`.
5. The CI runs a smoke test (`UIFOProblem` with a small budget) to verify your
   submission instantiates and produces a finite loss. Once it passes, the PR
   is queued for the next leaderboard evaluation.

A new submission replaces your previous one. The submission present in
`main` at the monthly deadline is what counts for that month. Your last
submission before the final deadline is what counts for the final leaderboard.

Read [docs/submission.md](docs/submission.md) for the full rules
(time budget, dependencies, disqualification criteria), and
[docs/scoring.md](docs/scoring.md) for how your run is scored.

---

## Improving the kit (external contributors)

Bug reports, documentation fixes, additional reference baselines, and
clarifications are all welcome.

### Reporting issues

Use GitHub Issues. A useful issue includes:

- What you ran (commit hash, command, seed).
- What you expected.
- What actually happened (full traceback, not a screenshot).
- The output of `python -c "import dfbench, differometor; print(dfbench.__version__, differometor.__version__)"`.

### Pull requests

- One PR per logical change.
- Match the existing style. We use `black` (line length 120) and basic
  `ruff` linting. Run them locally before pushing.
- Update or add docs in `docs/` if you change user-facing behaviour.
- Add a line to `CHANGELOG.md` under `## Unreleased`.

### Adding a new reference baseline

Reference baselines live in `learn2design/example_algorithms/` with a matching
runner in `learn2design/scripts/`. Keep both files short and self-contained;
the point of the kit is that participants can read everything in one sitting.

A new baseline PR should include:

- The algorithm file (single class, follows the contract).
- The runner script.
- A `docs/baselines.md` entry with a one-paragraph description and a small
  result table (e.g. mean loss on 5 fixed-seed `UIFOProblem` instances).

We are not aiming for a zoo of baselines, only methods that are pedagogically
useful or representative of an algorithm family will be merged.

---

## Development setup

```bash
git clone https://github.com/artificial-scientist-lab/Learn2Design-2026
cd Learn2Design-2026
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

The CI runs the same command on every PR. Smoke tests must pass before
merge; longer integration tests are run nightly.

---

## What we will not merge

- Submissions for the competition disguised as "improvements" to the starting
  kit. Submissions go through the submission flow above, not into
  `learn2design/`.
- Wholesale refactors of `learn2design/` without prior discussion. The kit is
  intentionally small.
- New mandatory dependencies. The participant-facing install must remain
  light.

If in doubt, open an issue before opening a PR.

---

## Security

If you find a security issue (sandbox escape, way to read private topologies,
etc.), do **not** open a public issue. Email
`jonathan.klimesch@uni-tuebingen.de` directly. We will respond within 72
hours.
