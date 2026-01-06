# Results directory

- `results/runs/` stores per-run outputs (logs, configs, metrics, checkpoints). Ignored by git.
- `results/tables/` stores aggregated tables/plots that we may commit for the paper.

Suggested run folder format:
results/runs/<timestamp>_<run_name>/
  - config.yaml
  - logs.txt
  - metrics.jsonl
  - summary.json
  - checkpoints/   (optional)
