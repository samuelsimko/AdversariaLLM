# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

AdversariaLLM is a toolkit for evaluating and comparing adversarial attacks (GCG, PAIR, AutoDAN, PGD, BEAST, Best-of-N, Crescendo, inpainting, etc.) on LLMs. The importable Python package is `adversariallm` (editable install from `pyproject.toml`). Three Hydra entry points orchestrate everything:

- `run_attacks.py` — run attacks and (optionally) judges. Main entry point.
- `run_judges.py` — re-judge already-generated runs, reusing saved completions.
- `run_sampling.py` — re-generate completions from saved attack inputs under a new generation config, then re-judge.

`strong_reject/` is a git submodule (requires `git submodule update --init`). `judges.py` at the repo root prepends it to `sys.path`.

## Environments

Two interchangeable setups. Do not mix them — pick one.

- **Pixi (recommended):** `pixi install --locked`, then `pixi run python ...` or `pixi shell`. Environment pins are in `pyproject.toml` under `[tool.pixi.*]` and `pixi.lock`. Python is pinned to 3.10.16; torch is CUDA 12.8.
- **pip/virtualenv:** `pip install -r requirements.txt && pip install -e .`

## Common commands

All commands assume the Pixi env is active (or prefix with `pixi run`).

```bash
# Attack a single model/dataset/attack slice. -m triggers Hydra multirun over idx.
python run_attacks.py -m \
    model=microsoft/Phi-3-mini-4k-instruct \
    dataset=adv_behaviors \
    datasets.adv_behaviors.idx="range(0,300)" \
    attack=gcg

# Sweep multiple attacks (one Hydra job per (attack, idx) pair)
python run_attacks.py -m attack=gcg,pair,autodan ...

# Override attack hyperparameters
python run_attacks.py -m attack=gcg attacks.gcg.num_steps=500 attacks.gcg.search_width=512

# Judge existing runs (no regeneration). classifier can be a judgezoo name or local:<name>.
python run_judges.py classifier=strong_reject
python run_judges.py classifier=local:ensemble

# Regenerate completions for already-saved attack inputs under a new generation config
python run_sampling.py  # edit conf/sampling.yaml filter_by first

# Remove orphaned DB rows / files
python purge_orphans.py [--dry-run] [--direction {db_only,files_only,both}]

# Tests
pytest -q                                      # full suite
pytest -q -m "not slow"                        # skip slow marker
pytest -q tests/test_attacks/test_direct.py    # single file
pytest -q tests/test_attacks/test_direct.py::test_name  # single test
```

Hydra multiruns land under `multirun/<date>/<time>/<i>/`. Attack result JSON files land under `outputs/<date>/<time>/<i>/run.json` (path is `save_dir=${root_dir}/outputs/`).

`root_dir` defaults to `${oc.env:PWD}` — override with `root_dir=/abs/path` or hard-code in `conf/paths.yaml`. Hydra launcher defaults to Slurm (`conf/hydra/launcher/a100h100.yaml`, partitions `gpu_a100,gpu_h100`); override with `hydra/launcher=...` or run with `hydra/launcher=basic` for local execution.

## Architecture

### Three-stage pipeline

1. **Attack.** `run_attacks.collect_configs` expands the Cartesian product of `(models × datasets × attacks)` and skips configs already present in the metadata DB. For each unique `RunConfig` it loads the model+dataset+attack once (avoiding reloads across the loop) and invokes `Attack.run(model, tokenizer, dataset) -> AttackResult`. Results are flushed via `log_attack`, which writes `run.json` and records a row in the metadata DB.
2. **Judge.** After attacks finish (or standalone via `run_judges.py`), judges score every saved completion and write scores back into the `run.json`, then append the score key to the DB row's `scored_by` array. `run_judges` skips files whose `scored_by` already contains the key — idempotent re-runs are safe.
3. **Resample.** `run_sampling.py` filters existing runs via `filter_by`, re-generates completions under a new `generation_config` (re-using saved `model_input_tokens`), re-judges them with the same classifiers, and writes a *new* `run.json` + DB row rather than mutating the old one.

### Registries via `from_name`

Attacks and datasets are registered by string name and dispatched at runtime:

- `Attack.from_name("gcg")` — see `adversariallm/attacks/attack.py`. New attacks must be added to the `match` statement (no decorator-based registration).
- `PromptDataset.from_name("adv_behaviors")` — uses a decorator registry (`@PromptDataset.register("...")`) in `adversariallm/dataset/prompt_dataset.py`.

Config files under `conf/attacks/attacks.yaml`, `conf/datasets/datasets.yaml`, `conf/models/models.yaml` key into these registries by matching name.

### Config layering

`conf/config.yaml` composes `attacks`, `datasets`, `models`, `paths`, and `hydra/launcher` via Hydra defaults. The top-level `generation_config` is interpolated into per-attack configs via `${generation_config}` (see `conf/attacks/attacks.yaml:_default`). Top-level `attack`, `dataset`, `model` fields select one registry entry to run; leaving them `null` runs all entries (non-underscored). `{attack,dataset,model}_overrides` dictionaries allow deep-merging of parameter overrides without editing the YAML.

### Result schema

`AttackResult` → list of `SingleAttackRunResult` (one per dataset index) → list of `AttackStepResult` per optimization step. Each step stores `model_input` (conversation), `model_input_tokens`, `model_completions: list[str]`, and `scores: dict[judge_name -> dict[field -> list[float]]]`. The distributional-attack design means `num_return_sequences > 1` produces a list of completions per step with one score per completion.

### Metadata backend

Defaults to **SQLite** at `outputs/runs.sqlite3` (`adversariallm/io_utils/database.py`); no server needed. Opt into MongoDB by exporting `ADVERSARIAL_DB_BACKEND=mongodb` + `MONGODB_URI` + `MONGODB_DB`. Override SQLite location with `ADVERSARIAL_SQLITE_PATH`. The code calls `get_mongodb_connection()` regardless of backend — it returns a Mongo-compatible wrapper around whichever backend is active.

Orphaned DB rows (pointing at deleted files) and orphaned files (missing from the DB) accumulate across failed runs — `delete_orphaned_runs()` (called automatically at the start of `run_judges`) and `purge_orphans.py` reconcile them.

### Judge backends

Two coexisting systems, selected by the classifier string (see `JUDGES.md`):

- **judgezoo** (default): plain names like `strong_reject`, `harmbench`. Stored score key = classifier name. Judge is called as `judge(modified_prompts)`.
- **local**: prefixed names like `local:ensemble`, `local:harmbench`, `local:strongreject`, `local:wildguard`, `local:jailjudge`, `local:gpt_oss`. Stored score key is the full `local:<name>`. Judge is called with three lists: `judge(harmful_prompts, attack_prompts, responses)`. Local judges score each completion twice — `score_without_jailbreak` (original harmful prompt) and `score_with_jailbreak` (the attacked prompt) — to separate context effects from completion harm.

Because the stored key differs (`strong_reject` vs `local:strongreject`), judgezoo and local runs of the "same" judge do not collide — a file can carry both.

### Determinism

All three entry points set `CUBLAS_WORKSPACE_CONFIG=":4096:8"` **before importing torch** and enable `torch.use_deterministic_algorithms(True, warn_only=True)`. Preserve this ordering when editing the top of these scripts.

## Defenses, experiments, JEPA stack

On top of the attack core there is a defense-training stack. Full walkthrough in `HOWTO_JEPA_AND_EXPERIMENTS.md`; key entry points:

- **`defenses/`** — standalone training scripts, one per defense. Each writes `lora_adapter/`, `manifest.json`, optional `jepa_predictor.pt`, `metrics.csv`, and a `READY` sentinel under its `--output_dir`. Current set: `align_jepa.py`, `harmful_jepa_cb.py`, `jepa_ce.py` (pluggable harm regularizer: `none`/`ce_floor`/`circuit_breaker`/`triplet`, plus `train_mode={full,predictor_only}` and optional `--init_adapter_path` for continued training), `ce_floor_base.py`, `ce_floor_align_jepa.py`, `ce_floor_refusal_attractor.py`, `honeypot_cb.py`, `velocity_collapse_cb.py`.
- **`experiments/run_experiment.py`** — YAML-first orchestrator that stitches `train → attack → benign_eval` per defense. Each stage is fingerprinted (SHA256 of stage config) so completed stages are reused across experiments by copy. Backends: `local`, `local_gpu` (sequential, respects `num_gpus`), `slurm` (sbatch with `--dependency=afterok` chains), `mock`. Outputs land under `runs/experiments/<name>/<defense.output_subdir>/`. Configs in `experiments/configs/*.yaml`.
- **`reverse_model/`** — LoRA-fine-tuned model that maps harmful behaviors → synthetic jailbreak prompts. Output dataset (`cb_train_reverse_prompts_5000_random_temp.jsonl`, ~164 MB) is the default `--pair_path` for JEPA defenses; each `(generated_prompts[i], true_prompt)` becomes a (adversarial, clean) pair. Holdout split: `reverse_model/holdout.json` (500 behaviors, never seen during training). Trained/sampled by `scripts/train_reverse_model.py` and `scripts/generate_reverse_prompts.py`.
- **`scripts/evaluate_jepa_guardrail.py`** — post-hoc eval of a trained JEPA defense. Loads the manifest + adapter + predictor, encodes texts at `align_layer` (and through the predictor head), and computes AUROC/TPR-at-FPR for centroid-based and predictor-based scores against benign vs. jailbreak prompts (UltraChat / WildJailbreak / reverse-model holdout).

### Conventions for defense scripts

Every defense in `defenses/` follows the same contract used by `experiments/run_experiment.py`:

- Accepts `--model`, `--output_dir`, plus its own hyperparameters.
- Writes `manifest.json` with at minimum: `schema_version`, `defense_name`, `base_model`, `adapter_type` (`"lora"` or `"none"`), `adapter_path` (relative path, or `null`), `training_completed`.
- Touches `READY` on success. The orchestrator uses this sentinel + the fingerprint hash to short-circuit reruns.
- For attack stages, `experiments/run_experiment.py` injects `+model_overrides.peft_path=<output_subdir>/<adapter_subdir>` (default `lora_adapter`). Set `attack_uses_adapter: false` on the defense YAML if the defense has no adapter (e.g. predictor-only training).
