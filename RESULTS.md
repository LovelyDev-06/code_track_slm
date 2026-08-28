# Results — Code Track

**Status:** scaffold only — no runs yet. Fill this in as `scripts/evaluate_all.py` produces
`logs/results/summary_table.csv`.

## Summary table

_Paste the output of `logs/results/summary_table.csv` here once you have runs._

| Strategy | Model | Dataset | Pass rate | Mean FLOPs/problem | Mean tokens/problem |
| --- | --- | --- | --- | --- | --- |
| greedy | qwen1_5b | humaneval | — | — | — |
| best_of_n | qwen1_5b | humaneval | — | — | — |
| self_consistency | qwen1_5b | humaneval | — | — | — |
| tree_search | qwen1_5b | humaneval | — | — | — |
| router | qwen1_5b | humaneval | — | — | — |

(repeat for qwen7b and mbpp)

## FLOPs vs. accuracy

See `logs/results/flops_vs_accuracy.png` after running `evaluate_all.py`.

## Known limitations to flag before citing numbers in the paper

- Small `--limit` sample sizes during development runs will have noisy pass rates —
  rerun without `--limit` (or with a larger one) before reporting final numbers.
- The learned verifier and router are trained on MBPP's `train` split by default (see
  `scripts/train_verifier.py` and `scripts/train_router.py` docstrings) — if you retrain
  on a different split, note that here, since it affects how directly comparable the
  verifier/router numbers are across datasets.
- `router.strategies_available` in `configs/config.yaml` excludes `self_consistency` by
  default (only greedy / best_of_n / tree_search) to keep router training cost down —
  note this if the paper claims the router chooses among "all six strategies".
- Tree search's `heuristic_score` fallback (used when no trained verifier is present) is
  a rough proxy — results using it should be flagged as such and ideally rerun once
  `train_verifier.py` has been run.

## Next steps

- [ ] Full (non-`--limit`) runs of all six strategies on both models × both benchmarks
- [ ] Cross-check a sample of `passed=True` results manually to make sure `run_tests` isn't
      producing false positives (e.g. from an empty test file silently "passing")
- [ ] Compare router's accuracy-per-FLOP against always-using-best-single-strategy baseline
