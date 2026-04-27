# How to use JEPA defenses, the reverse model, and the experiment pipeline

This is a practical guide to the three newer pieces of AdversariaLLM that sit on top of the attack/judge/sampling core described in `CLAUDE.md`:

1. **`reverse_model/`** — a LoRA-fine-tuned LLM trained to map *harmful responses → candidate jailbreak prompts*, plus the JSONL dataset of generated prompts that this produces.
2. **`defenses/align_jepa.py` and `defenses/harmful_jepa_cb.py`** — JEPA-style (Joint Embedding Predictive Architecture) defense training scripts that use the reverse-model output (and benign data) to fine-tune a defended model.
3. **`experiments/run_experiment.py`** — a YAML-first experiment launcher that stitches *train → attack → benign_eval* stages together, with fingerprint-based reuse and pluggable backends (local / local_gpu / slurm / mock).

> If you only want to run attacks against an off-the-shelf model, you do not need any of this — use `run_attacks.py` / `run_judges.py` / `run_sampling.py` directly (see `CLAUDE.md`). Reach for this stack when you want to *train a defended model and attack it*.

---

## 1. Big picture

```
 data/circuit_breakers_train.json
          │
          ▼
 ┌─────────────────────────────────┐
 │ scripts/train_reverse_model.py  │  (LoRA on a base LLM)
 │   → reverse_model/lora/         │
 └─────────────────────────────────┘
          │
          ▼
 ┌────────────────────────────────────────┐
 │ scripts/generate_reverse_prompts.py    │  (100s of prompts/behavior)
 │   → reverse_model/cb_train_reverse_*   │
 │     _prompts_5000_random_temp.jsonl    │
 └────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────┐     (benign data from ultrachat_200k)
 │ defenses/align_jepa.py          │◄────────────────────────────
 │   → runs/.../lora_adapter/      │
 └─────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────┐
 │ run_attacks.py                                  │
 │   +model_overrides.peft_path=.../lora_adapter   │
 │   → outputs/.../run.json (+ DB row)             │
 └─────────────────────────────────────────────────┘

 The whole chain is orchestrated end-to-end by:
    experiments/run_experiment.py --config <yaml> --backend <...>
```

- `harmful_jepa_cb.py` is a sibling to `align_jepa.py`: same slot in the pipeline (produces a LoRA adapter), but a different training objective (representation-level defense using circuit-breaker data).
- Once a defense has produced a LoRA adapter, the rest of the pipeline (attacks + judges + benign-capability eval) is the stock AdversariaLLM machinery with `peft_path` pointing at the adapter.

---

## 2. The reverse model

### What it is

A small LoRA fine-tune of a base instruction model trained to do the **reverse** of normal chat: given an already-written harmful response, produce plausible user prompts that could have elicited it. It is *not* itself an attack — it is a dataset generator. Its output is used by `align_jepa.py` to construct (jailbreak-prompt, behavior) pairs for the alignment objective.

### Contents of `reverse_model/`

- `lora/` — trained LoRA adapter (the "reverse model").
- `checkpoint-1990/`, `checkpoint-2000/` — intermediate training checkpoints.
- `holdout.json` — 500-behavior held-out split created during training.
- `cb_train_reverse_prompts_5000_random_temp.jsonl` — the main generated dataset (5000 circuit-breaker behaviors × many sampled prompts across random temperatures).
- `cb_train_reverse_prompts_5000_random_temp.jsonl.events.jsonl` — event log (progress / temperature schedule).
- `cb_train_reverse_prompts_5000_random_temp_backups/` — progressive snapshots from `--backup-every`.
- `cb_train_reverse_prompts.jsonl`, `cb_train_reverse_prompts_test.jsonl` — earlier / smaller runs.
- `test_generations.json` — a few generations for eyeballing quality.

### Record shape (per line in the JSONL)

```json
{
  "index": 0,
  "record": { ...original circuit_breakers record... },
  "response": "...harmful response text...",
  "true_prompt": "...the original prompt that elicited the response...",
  "generated_prompts": ["...", "...", "..."],
  "generated_prompts_by_temperature": { "0.4": ["..."], "0.7": ["..."] }
}
```

`align_jepa.py` consumes these files and pairs each `generated_prompts[i]` with the `true_prompt` or behavior text of the same record (that pairing is the supervision signal).

### Retraining / regenerating from scratch

Two scripts plus one convenience shell wrapper:

```bash
# 1) Train the reverse model
python scripts/train_reverse_model.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --cb_path data/circuit_breakers_train.json \
  --output_dir reverse_model \
  --cb_limit 5000 \
  --ultra_samples 20000 \
  --holdout 500 \
  --max_steps 2000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4

# 2) Sample jailbreak-prompt candidates from the trained reverse model
python scripts/generate_reverse_prompts.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter-dir reverse_model/lora \
  --cb-path data/circuit_breakers_train.json \
  --num-generations 100 \
  --random-temperature-count 5 \
  --temperature-min 0.0 --temperature-max 3.0 \
  --top-p 0.95 --top-k 20 \
  --output-path reverse_model/cb_train_reverse_prompts_5000_random_temp.jsonl \
  --backup-every 10
```

End-to-end wrapper (honors env-var overrides for every flag):

```bash
# Run both stages with defaults:
bash scripts/run_reverse_model_pipeline.sh

# Skip training and reuse an existing adapter:
SKIP_TRAIN=1 ADAPTER_DIR=reverse_model/lora bash scripts/run_reverse_model_pipeline.sh
```

Relevant env vars: `PYTHON_BIN`, `CB_PATH`, `REVERSE_DIR`, `ADAPTER_DIR`, `SKIP_TRAIN`, `BASE_MODEL`, `CB_LIMIT`, `TRAIN_MAX_STEPS`, `TRAIN_BATCH_SIZE`, `GEN_LIMIT`, `NUM_GENERATIONS`, `RANDOM_TEMPERATURE_COUNT`, `TEMPERATURE_MIN/MAX`, `OUTPUT_PATH` — see the script for the complete list.

If you already have the bundled `cb_train_reverse_prompts_5000_random_temp.jsonl` and just want to train defenses, **skip this section entirely**.

---

## 3. JEPA defenses

Both defenses live in `defenses/` and share the same contract:

- Inputs: a base model id + some data file(s).
- Output: a directory containing a `lora_adapter/` subdirectory (plus `manifest.json`, `metrics.csv`, `metrics.png`, and a `READY` marker on success).
- Consumption: downstream `run_attacks.py` loads the base model and applies the LoRA via `+model_overrides.peft_path=<output_dir>/lora_adapter`.

### 3.1 `defenses/align_jepa.py` — LLM-JEPA-style alignment

**Idea.** With a tied-weights LLM-JEPA predictor, `Pred(Enc(jailbreak))` is realized by appending a `[PRED]` special token to the jailbreak prompt and reading the last-token hidden state. `Enc(behavior)` is the last-token hidden state of the *behavior text* (intent), not the full harmful response. The loss pulls `Pred(Enc(jailbreak))` toward `Enc(behavior)` (cosine or L2), while a benign CE loss and optional benign-KL term preserve general capability.

Total loss:
```
loss = w_benign * CE(benign)        # preserve general LM
     + w_align  * d(pred, target)   # JEPA alignment (cosine or L2)
     + w_benign_kl * KL(adapter || base)   # optional: stay close to base on benign text
```

**Minimum run** (from `run_aligned_jepa.sh`):

```bash
python defenses/align_jepa.py \
  --model Qwen/Qwen3-8B \
  --reverse_path reverse_model/cb_train_reverse_prompts_5000_random_temp.jsonl \
  --output_dir runs/align_jepa_qwen3_8b_tied
```

**Common flags** (see `defenses/align_jepa.py` around lines 511–539 for the full list):

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `Qwen/Qwen3-8B` | Base HF model id. |
| `--reverse_path` | *required* | JSONL from the reverse model. |
| `--output_dir` | `./align_jepa_qwen` | Where to save the LoRA + artifacts. |
| `--ultrachat_samples` | 5000 | Benign samples from ultrachat_200k. |
| `--align_limit` | 5000 | Max (jailbreak, behavior) pairs. |
| `--max_length` | 256 | Sequence length. |
| `--batch_size` / `--grad_accum` | 4 / 2 | Per-device batch, accumulation. |
| `--num_max_steps` | 1500 | Training steps. |
| `--lr` | 2e-4 | Learning rate. |
| `--w_benign` / `--w_align` / `--w_benign_kl` | 1.0 / 1.0 / 0.0 | Loss weights. |
| `--num_pred_tokens` | 1 | Number of `[PRED]` tokens appended. |
| `--align_layer` | -1 | Hidden layer for the JEPA reps. |
| `--align_metric` | `cosine` | `cosine` or `l2`. |
| `--lora_r` / `--lora_alpha` / `--lora_dropout` | 32 / 16 / 0.05 | LoRA hyperparams. |
| `--target_modules` | `q_proj,v_proj` | Modules to adapt. |

Outputs written under `--output_dir`: `lora_adapter/`, `metrics.csv`, `metrics.png`, `manifest.json`, `hparams.json`, and `READY`.

### 3.2 `defenses/harmful_jepa_cb.py` — representation-level JEPA on circuit breakers

Same output shape as `align_jepa.py` but the loss operates on representations across multiple layers (default layers 20–30) and mixes a harm-JEPA term with a KL anchor to the base model on benign data. Arguments (see `defenses/harmful_jepa_cb.py` around lines 288–304):

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `meta-llama/Meta-Llama-3-8B-Instruct` | Base model. |
| `--cb_path` | *required* | Circuit-breaker data file. |
| `--output_dir` | `./harmful_jepa` | Artifact root. |
| `--rep_layers` | `20..30` | Layers used for the representation loss. |
| `--w_harm_jepa` / `--w_kl` | 2.0 / 1.0 | Loss weights. |
| `--predictor_type` | `identity` | Predictor head style. |
| `--ultrachat_samples` / `--limit_cb` | 5000 / 5000 | Data sizes. |
| `--lora_r` / `--lora_alpha` / `--lora_dropout` | 32 / 64 / 0.0 | LoRA hyperparams. |

Minimal run:

```bash
python defenses/harmful_jepa_cb.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --cb_path data/circuit_breakers_train.json \
  --output_dir runs/harmful_jepa_llama3
```

### 3.3 Other defense scripts in `defenses/`

Same contract as the two above (LoRA adapter + manifest + `READY`), different objectives. Use any of these as a `script:` in an experiment YAML.

| Script | Idea |
|---|---|
| `defenses/jepa_ce.py` | JEPA per-position predictor + a *pluggable* harmful regularizer (`--harm_regularizer ce_floor|circuit_breaker|triplet|none`) and a CE-loss term on benign data. Supports `--train_mode predictor_only` (freeze backbone, train only the predictor) and `--init_adapter_path` to continue training from an existing adapter. |
| `defenses/ce_floor_base.py` | CE-floor only baseline: enforce a minimum CE loss on harmful prompts so the model cannot fluently complete them. |
| `defenses/ce_floor_align_jepa.py` | CE-floor + `align_jepa`-style alignment combined. |
| `defenses/ce_floor_refusal_attractor.py` | Representation-level defense with a learned "refusal attractor" subspace, plus margin/orthogonality terms. |
| `defenses/honeypot_cb.py` | Honeypot-based defense: builds honeypot representations and trains margin losses (ref > honeypot > harmful in rep-space). |
| `defenses/velocity_collapse_cb.py` | Velocity-based: collapse the hidden-state velocity of harmful prompts while preserving benign velocity. |

After training, `scripts/evaluate_jepa_guardrail.py --run_dir <output_dir>` produces AUROC / TPR-at-FPR for predictor- and centroid-based guardrails over benign vs. jailbreak prompts (UltraChat / WildJailbreak / reverse-model holdout). Reads `manifest.json`; `adapter_path` may be `null` (e.g. predictor-only training), in which case it loads the base tokenizer and skips the LoRA wrap.

### 3.4 Attacking a trained defense by hand

Once any defense has written a `lora_adapter/`, you can attack it directly with `run_attacks.py`:

```bash
python run_attacks.py -m \
  model=meta-llama/Meta-Llama-3-8B-Instruct \
  dataset=adv_behaviors \
  datasets.adv_behaviors.idx="range(0,10)" \
  attack=direct,gcg \
  +model_overrides.peft_path=runs/align_jepa_qwen3_8b_tied/lora_adapter
```

`run_judges.py` then scores the resulting `outputs/.../run.json` files as usual.

---

## 4. The experiment pipeline

### 4.1 What `experiments/run_experiment.py` does

Given one YAML config, it:

1. Loads the experiment config plus the main registries (`conf/attacks/attacks.yaml`, `conf/datasets/datasets.yaml`, `conf/models/models.yaml`).
2. Expands each *pipeline* into an ordered list of stages (`train`, `attack`, `benign_eval`).
3. For every stage, builds a command, computes a **fingerprint** over the relevant config, and checks `<stage_dir>/fingerprint.json` + `READY`:
   - If the same fingerprint already completed, it is skipped (or copied from another location if found elsewhere under `output_root`).
4. Wraps each command in `experiments/execute_job.py` (which writes `command.txt`, `status.json`, and `READY` / `FAILED`).
5. Submits to the chosen backend:
   - `local` — run now, in-process.
   - `local_gpu` — sequential, one GPU at a time (good default on a single box).
   - `slurm` — `sbatch` each job, chain attack/benign jobs on the train job via `--dependency=afterok:<train_job_id>`.
   - `mock` — print the command without running.
6. Appends one JSON record per submitted job to `<output_root>/<experiment_name>/jobs.jsonl` and writes `submission_manifest.json`.

Attack stages call `run_attacks.py` (the library entry point). If the defense has a `script`, the attack stage additionally sets `+model_overrides.peft_path=<defense_output>/lora_adapter` so the attack sees the defended model. Benign eval calls `benign_capabilities/run_benign_eval.py`.

### 4.2 Running an existing experiment

```bash
# Single machine, sequential:
python experiments/run_experiment.py \
  --config experiments/configs/align_jepa_qwen3_long_inpainting.yaml \
  --backend local_gpu

# Slurm cluster:
python experiments/run_experiment.py \
  --config experiments/configs/align_jepa_qwen3_long_inpainting.yaml \
  --backend slurm

# Dry-run — just print what would be submitted:
python experiments/run_experiment.py \
  --config experiments/configs/align_jepa_qwen3_long_inpainting.yaml \
  --backend mock
```

Outputs land under `runs/experiments/<experiment_name>/`:

```
runs/experiments/<experiment_name>/
├── logs/                              # stdout/stderr per job
├── submission_manifest.json
├── jobs.jsonl
└── <defense.output_subdir>/           # training artifacts (lora_adapter/, metrics...)
    ├── fingerprint.json
    ├── status.json  READY|FAILED
    ├── lora_adapter/
    ├── manifest.json  metrics.csv  metrics.png
    ├── attacks/<pipeline>/<attack_name>/    # run_attacks.py outputs
    │   ├── fingerprint.json  READY|FAILED
    │   ├── outputs/           # run.json lives here
    │   └── hydra/ hydra_multirun/
    └── benign_eval/<pipeline>/<benign_eval>/
        └── results/  READY|FAILED
```

### 4.3 Bundled example configs

| Config | What it does |
|---|---|
| `experiments/configs/library_pipeline_example.yaml` | Minimal template: one base-model pipeline (no training) + one honeypot-CB defense, two smoke attacks, one benign eval. Good starting point. |
| `experiments/configs/align_jepa_qwen3_long_inpainting.yaml` | Full `align_jepa.py` training on Qwen3-8B, then 6 attacks (direct, prefilling, bon, gcg, soft_prompt, inpainting) and a gsm8k+mmlu smoke eval. |
| `experiments/configs/harmful_jepa_dual_family.yaml` | Same plan as above, but trains both Llama-3-8B and Qwen3-8B with `harmful_jepa_cb.py`. |
| `experiments/configs/velocity_inpainting_dual_family.yaml` | Same plan but with the `velocity_collapse_cb.py` defense. |
| `experiments/configs/triplet_attack_suite.yaml` | No training — just attacks against a pre-specified set of models. |

There is also legacy JSON support (`experiments/*.json`) and two generator scripts:

```bash
python experiments/generate_experiment.py                 # writes an example JSON config
python experiments/generate_triplet_attack_experiment.py  # writes triplet_attack_suite.yaml
```

### 4.4 Anatomy of an experiment YAML

```yaml
meta:
  experiment_name: my_experiment          # becomes the output subdir name
  output_root: runs/experiments           # base dir for everything

runtime:
  python_bin: /workspace/AdversariaLLM/venv/bin/python    # optional
  venv_activate: /workspace/AdversariaLLM/venv/bin/activate  # optional
  env_file: .env                                          # optional

cluster:                                  # consumed by slurm backend
  partition: gpu_a100
  account: my_account
  gres: gpu:1
  num_gpus: 1                             # used by local_gpu backend
  time_train: "16:00:00"
  time_attack: "06:00:00"
  time_benign: "02:00:00"

default_attack_dataset: adv_behaviors     # used if an attack omits `dataset`

classifiers:                              # judges to run after every attack
  - local:strongreject
  - local:harmbench

# ---- Registries inside the experiment ----
models:
  qwen3_8b:
    from_registry: Qwen/Qwen3-8B          # reference conf/models/models.yaml
  custom_model:                           # or define inline
    id: org/model
    tokenizer_id: org/model               # optional
    dtype: bfloat16                       # optional

datasets:
  my_subset:
    name: adv_behaviors                   # PromptDataset.from_name key
    idx: [0, 1, 2, 3, 4]                  # subset the dataset
    seed: 0
    shuffle: true

defenses:
  my_defense:
    script: defenses/align_jepa.py        # set null to skip training
    base_model: qwen3_8b                  # which entry in `models` to fine-tune
    output_subdir: my_defense             # folder under experiment output
    data:                                 # optional: shorthand for --cb_path etc.
      cb_path: data/circuit_breakers_train.json
    train_args:                           # passed as --flag value to the script
      reverse_path: reverse_model/cb_train_reverse_prompts_5000_random_temp.jsonl
      num_max_steps: 4000
      batch_size: 8
      w_align: 1.0
      w_benign: 0.5
      w_benign_kl: 0.2

attacks:
  direct_on_subset:
    attack: direct                        # registry name from conf/attacks/attacks.yaml
    dataset: my_subset
  gcg_fast:
    attack: gcg
    dataset: my_subset
    attack_overrides:                     # deep-merged into the attack config
      num_steps: 40
      search_width: 64
      batch_size: 32
    classifiers:                          # optional: override top-level classifiers
      - local:strongreject

benign_evals:
  gsm8k_smoke:
    tasks: gsm8k,mmlu
    limit: 200

pipelines:
  my_pipeline:
    - stage: train
      defense: my_defense
    - stage: attack
      defense: my_defense
      attacks: [direct_on_subset, gcg_fast]
    - stage: benign_eval
      defense: my_defense
      benign_eval: gsm8k_smoke
```

### 4.5 Backend notes

- **`local_gpu`** runs jobs one-at-a-time on the current box, waiting between them. It honors `cluster.num_gpus`. This is the easiest way to run a full experiment on a single workstation.
- **`slurm`** submits each stage as a separate `sbatch` job and chains attack/benign jobs on the train job via `--dependency=afterok:<id>`. The default Hydra launcher for inner `run_attacks.py` runs is overridden to `basic` when the experiment runner is already on a Slurm node — the outer experiment runner is the scheduler, the inner attack is a plain process.
- **`mock`** prints the wrapped commands (useful for `--backend mock | less`).
- **`local`** runs in-process; avoid for anything non-trivial.

### 4.6 Fingerprinting and reuse

Every stage writes `fingerprint.json` containing the resolved stage config. On subsequent runs:

- If the same `stage_dir` already has a matching `fingerprint.json` and `READY`, the stage is skipped.
- Otherwise the runner scans every `fingerprint.json` under `output_root` and, if it finds a match elsewhere, **copies** that completed stage into the new location and records the reuse in `jobs.jsonl` as `reused_from`.

This makes it cheap to spin up variants of an experiment that share some stages — e.g., rerunning the same training with a new attack list.

---

## 5. Building a new experiment end-to-end

Scenario: *"I have a new defense idea. I want to train it on Llama-3-8B using the existing reverse-prompt dataset, attack it with GCG and BoN on 20 behaviors, and check it still gets passable gsm8k."*

**Step 1.** Write the trainer (or reuse one). It must:

- Accept `--model`, `--output_dir`, plus whatever else you need as `--flag value` CLI args.
- Write a LoRA adapter under `<output_dir>/lora_adapter/`.
- Touch `<output_dir>/READY` on success (optional — the harness re-touches it, but doing it inside the script is polite).

`defenses/align_jepa.py` and `defenses/harmful_jepa_cb.py` are both good models for this contract.

**Step 2.** Add a config at `experiments/configs/my_new_defense.yaml`:

```yaml
meta:
  experiment_name: my_new_defense
  output_root: runs/experiments

runtime:
  python_bin: /workspace/AdversariaLLM/venv/bin/python

cluster:
  partition: tamper_resistance
  account: replace_me
  gres: gpu:1
  num_gpus: 1
  time_train: "08:00:00"
  time_attack: "04:00:00"
  time_benign: "01:00:00"

default_attack_dataset: adv_small

classifiers:
  - local:strongreject

models:
  llama3_8b:
    from_registry: meta-llama/Meta-Llama-3-8B-Instruct

datasets:
  adv_small:
    name: adv_behaviors
    idx: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    seed: 0
    shuffle: true

defenses:
  my_new_defense:
    script: defenses/my_new_defense.py
    base_model: llama3_8b
    output_subdir: my_new_defense
    train_args:
      reverse_path: reverse_model/cb_train_reverse_prompts_5000_random_temp.jsonl
      num_max_steps: 2000
      batch_size: 4

attacks:
  direct_smoke:
    attack: direct
    dataset: adv_small
  gcg_short:
    attack: gcg
    dataset: adv_small
    attack_overrides:
      num_steps: 40
      search_width: 64
  bon_short:
    attack: bon
    dataset: adv_small
    attack_overrides:
      num_steps: 200
      sigma: 0.4

benign_evals:
  gsm8k_smoke:
    tasks: gsm8k
    limit: 200

pipelines:
  my_new_defense:
    - stage: train
      defense: my_new_defense
    - stage: attack
      defense: my_new_defense
      attacks: [direct_smoke, gcg_short, bon_short]
    - stage: benign_eval
      defense: my_new_defense
      benign_eval: gsm8k_smoke
```

**Step 3.** Dry-run to check command strings:

```bash
python experiments/run_experiment.py --config experiments/configs/my_new_defense.yaml --backend mock
```

**Step 4.** Run for real:

```bash
python experiments/run_experiment.py --config experiments/configs/my_new_defense.yaml --backend local_gpu
# or --backend slurm on a cluster
```

**Step 5.** Inspect:

```
runs/experiments/my_new_defense/
├── jobs.jsonl                          # one line per submitted/reused stage
├── logs/                               # stdout/stderr
└── my_new_defense/
    ├── lora_adapter/                   # trained weights
    ├── metrics.csv / metrics.png
    ├── manifest.json
    └── attacks/my_new_defense/
        ├── direct_smoke/outputs/…/run.json
        ├── gcg_short/outputs/…/run.json
        └── bon_short/outputs/…/run.json
```

Every `run.json` carries the normal `scored_by` list and per-judge scores, so standard tools (`run_judges.py`, `scripts/analyze_run_asr.py`, the SQLite metadata DB) all work.

**Step 6 (optional).** Re-judge with more classifiers later — scoring is idempotent and keyed by judge name:

```bash
python run_judges.py classifier=local:harmbench
```

---

## 6. Quick-reference cheatsheet

```bash
# Generate the reverse-prompt dataset (one-time, ~hours):
bash scripts/run_reverse_model_pipeline.sh

# Train align-JEPA by itself:
bash run_aligned_jepa.sh

# Run a full train → attack → benign_eval experiment locally:
python experiments/run_experiment.py \
  --config experiments/configs/align_jepa_qwen3_long_inpainting.yaml \
  --backend local_gpu

# Same on Slurm:
python experiments/run_experiment.py \
  --config experiments/configs/align_jepa_qwen3_long_inpainting.yaml \
  --backend slurm

# Attack an already-trained LoRA defense directly:
python run_attacks.py -m \
  model=Qwen/Qwen3-8B \
  dataset=adv_behaviors \
  datasets.adv_behaviors.idx="range(0,10)" \
  attack=gcg \
  +model_overrides.peft_path=runs/align_jepa_qwen3_8b_tied/lora_adapter

# Re-judge with a different classifier (idempotent):
python run_judges.py classifier=local:harmbench

# Resample completions under a new generation config:
python run_sampling.py          # edit conf/sampling.yaml filter_by first
```

---

## 7. Gotchas

- **Don't mix environments.** Pick pixi *or* the `venv/` in the repo. The experiment YAMLs point at `runtime.python_bin`; make sure that interpreter has the right packages.
- **`align_jepa.py` adds the `[PRED]` special token** and resizes the embedding matrix. If you swap the tokenizer after training you will silently break the alignment objective. Always load the tokenizer saved alongside `lora_adapter/`.
- **`w_benign_kl` needs a LoRA-wrapped model** so `disable_adapter()` can produce the base-model logits. If you remove LoRA you must also set `--w_benign_kl 0`.
- **Fingerprints pin data-file *paths***, not their contents. If you change the content of `reverse_model/*.jsonl` without renaming it, previous runs will still be marked completed. Rename or bump the `output_subdir` to force a retrain.
- **`peft_path` override is additive** (`+model_overrides.peft_path=…`). If you already set `model_overrides.peft_path` elsewhere in your Hydra config, the experiment runner's override will conflict — pick one source of truth.
- **Slurm dependency is `afterok`**, so a failed training stage blocks its attacks forever. Either fix and re-submit, or delete the train `FAILED` marker and the downstream `fingerprint.json`s before retrying.
- **The root-level scripts `judges.py` and `new_judges.py`** and the files `test.py`, `test_triplet.py`, `tea_debug.log`, `tmp`, and `vc/` in the working tree are scratch / WIP and unrelated to this pipeline. Don't import them from experiment code.
