# Resumable checkpoints: ` .json` + `.csv`

This project now uses PyTorch ` .json` files for checkpoints/progress and `.csv` files for tabular experiment results. JSON/JSONL result files are no longer produced by the run scripts.

## What is saved

For each evaluation run:

- `checkpoints/<strategy>_<model>_<dataset>_<split>_limitN .json` — completed results + FLOP ledger + run metadata.
- `logs/<strategy>_<model>_<dataset>_<split>_limitN.csv` — final/current tabular results.

For router training:

- `checkpoints/router_labels_.. .json` — per-problem strategy labels/progress.
- `checkpoints/router .json.train .json` — model + optimizer + epoch state, so training resumes from the next epoch.
- `checkpoints/router .json` — final router weights used for inference.

## Resume after shutdown

Run the exact same command again. The script loads the local ` .json` checkpoint and skips completed problem IDs.

Example:

```bash
python scripts/run_best_of_n.py --model qwen1_5b --dataset mbpp --limit 470
```

If the machine stops after problem 183, the next run skips 1–183 and continues at 184.

If the local checkpoint is missing and Hub pushing is enabled, the script attempts to restore the matching ` .json` checkpoint from the configured Hugging Face repository.

## Avoiding data loss

- A checkpoint is written after every completed problem.
- Checkpoint writes use an atomic temporary-file replacement, so a shutdown during a write is much less likely to corrupt the previous checkpoint.
- Hugging Face upload frequency is controlled by `hub.push_every_n_problems` in `configs/config.yaml`.
- Use `--no_push` for local-only runs.

## Evaluation summary

Run:

```bash
python scripts/evaluate_all.py
```

It reads the strategy `.csv` files, creates `logs/results/summary_table.csv`, and generates the Pass Rate vs Test-Time Compute plot.

## Updated implementation notes
- Router remains a **Learned Latent Router** and now learns over four strategies: `greedy`, `best_of_n`, `self_consistency`, and `tree_search`.
- `self_consistency` is preserved as an independent strategy with repeated sampling and AST-normalized consensus voting.
- Tree Search now uses a **small LLM-based step judge** for intermediate branch scoring/pruning; final verification remains AST parsing plus benchmark execution.
- Progress checkpoints are JSON and router weights are SafeTensors; no `.pt` checkpoints are used.
