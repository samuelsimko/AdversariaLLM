## Experiment Pipeline

`experiments/run_experiment.py` is now a YAML-first launcher for the existing library pipeline.

What changed:
- Attack stages run `run_attacks.py` instead of the old `attacks/run_attack.py` path.
- Attack configs should use library attacks from `conf/attacks/attacks.yaml`.
- Defended models are attacked by loading the base model plus the trained LoRA adapter via `model_overrides.peft_path`.
- Each submitted stage writes `fingerprint.json`, `status.json`, `command.txt`, and `READY`/`FAILED` in its stage directory.
- Submission metadata is appended to `runs/experiments/<experiment>/jobs.jsonl`.

Run locally:

```bash
python experiments/run_experiment.py --config experiments/configs/library_pipeline_example.yaml --backend local_gpu
```

Run on Slurm:

```bash
python experiments/run_experiment.py --config experiments/configs/library_pipeline_example.yaml --backend slurm
```

Config notes:
- `models` may reference the main registry with `from_registry`, or define inline model params.
- `attacks` should prefer `attack: <registry-name>` plus `attack_overrides`.
- `datasets` may reference the main registry with `from_registry`; if omitted, attacks default to `adv_behaviors`.
- `benign_evals` still use `benign_capabilities/run_benign_eval.py`.
- Legacy JSON experiment files still load, but legacy attack names that are not present in the library attack registry now fail with a clear error.
