# Setup Guide — AdversariaLLM 20-Cell Headline Rerun

Read this end-to-end before submitting anything. Estimated setup time: **30–45
minutes** (most of it waiting for `pip install` and `huggingface-cli login`).

## What you're going to run

A 20-cell sweep: 16 trained defenses (matched-pair PRA vs no-PRA design across
4 harm regularizers × 2 backbones) + 4 published reference defenses, attacked
with a 7-attack suite (soft_prompt, direct, prefilling, human_jailbreaks,
or_bench, bon, inpainting) + benign evals (gsm8k_200, mmlu_full).

One sbatch job per cell, one node per sbatch (4 GPUs per node). All results
sync to a private HuggingFace dataset repo as each stage completes.
Total wall-clock ≈ **3 hours** on 16 nodes (LAT cells are the long pole at
~2 h training + 1 h attacks/eval).

---

## 1. Clone & submodules

```bash
git clone <YOUR_REPO_URL> /workspace/AdversariaLLM
cd /workspace/AdversariaLLM
git submodule update --init --recursive
```

Two submodules will be cloned:
- `strong_reject/` — required by the `local:strongreject` judge (one of two
  judges used to score attacks). `judges.py` at the repo root prepends
  `strong_reject/` to `sys.path`. **If this submodule is missing, judging
  silently produces NaN scores.**
- `latent-adversarial-training/` — reference copy of the upstream LAT codebase
  (Sheshadri et al.). Read-only; not imported at runtime.

Verify:
```bash
ls strong_reject/strong_reject/                  # should have judge code
ls latent-adversarial-training/latent_at/        # should have lat_methods.py
```

---

## 2. Python environment

Pinned to Python 3.10. **Do not install vllm** — it pulls a torch version
incompatible with our CUDA stack.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e . --no-deps           # editable install of the adversariallm package
pip install lm-eval                  # required for benign evals (gsm8k, mmlu)
```

Check torch + CUDA work:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# expect: 2.7.1+cu126 True 4   (or whatever the node's GPU count is)
```

If you ever need to nuke the env and start over, just `rm -rf .venv` and
re-run the block above — there's no other state.

---

## 3. HuggingFace credentials & dataset access

The training data pulls **two gated HF datasets**: `allenai/wildguardmix`
and `allenai/wildjailbreak`. You must accept both licenses **before** running
the data-build step or it will 403 silently.

1. **Accept the licenses** (one click each, on a logged-in HF account):
   - https://huggingface.co/datasets/allenai/wildguardmix
   - https://huggingface.co/datasets/allenai/wildjailbreak

2. **Mint an HF token with write access** to your account (or your team org):
   https://huggingface.co/settings/tokens → "Create new token" → role: "Write".

3. **Create the private dataset repo** that all sync will push into:
   ```bash
   huggingface-cli login          # paste your write token
   python -c "from huggingface_hub import HfApi; HfApi().create_repo(
       'YOUR_USER_OR_ORG/headline-rerun', repo_type='dataset', private=True, exist_ok=True)"
   ```
   Replace `YOUR_USER_OR_ORG` with your HF username or your team's HF org.
   You'll use this string as `HF_REPO` below.

---

## 4. Build training data

This downloads WildGuardMix + WildJailbreak (both ~few GB cached locally),
filters and aligns them, and writes three local files to `data/`:

```bash
python scripts/build_wildjailbreak_data.py
```

Output:
```
data/wildjailbreak_pairs.jsonl     # ~10k (vanilla, adversarial) pairs for PRA
data/wildjailbreak_harmful.json    #  5k (prompt, harmful_response) for harm-CE
data/wildjailbreak_benign.jsonl    #  5k (prompt, helpful_response) for benign retain
```

Takes 1–5 minutes if HF datasets are cached, 10–20 minutes on first download.
If you see `gated dataset` or `403` errors, you didn't accept the licenses
in step 3.1 — go do that, then re-run.

---

## 5. The `.env` file

Create `.env` at the repo root. Each sbatch sources this on the node before
running anything. Required:

```bash
# Copy this block into .env, fill in YOUR_USER_OR_ORG and your token:
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export HF_REPO="YOUR_USER_OR_ORG/headline-rerun"
```

Optional:

```bash
# Enable / disable HF push at runtime. Defaults to enabled.
export HF_SYNC_ENABLED=1

# Where to find the Why-Probe-Fails repo for the probe stage.
# Only needed if you'll re-enable probing in the YAML's pipelines.
export WPF_ROOT="/workspace/Why-Probe-Fails"

# If you have OpenAI API access for any judge that calls a remote model.
# (The default judge is local:strongreject; doesn't need this.)
export OPENAI_API_KEY="sk-..."
```

**Do not commit `.env` to git.** It's in `.gitignore`; double-check before
pushing anything.

---

## 6. Cluster-specific customization

The slurm template (`cluster_scripts/cell.slurm.template`) ships with these
defaults — tweak via env vars at submit time, or edit the template directly:

| Slurm header | Default | Env override | Where to find your value |
|---|---|---|---|
| `--account` | `infra01` | `ACCOUNT=...` | `sacctmgr show user $USER format=defaultaccount` or ask cluster admin |
| `--partition` | `normal` | `PARTITION=...` | `sinfo -s` to list partitions |
| `--time` | `04:00:00` | `TIME=HH:MM:SS` | leave at 4h unless your partition has a stricter cap |
| `--gpus-per-node` | `4` | `GPUS_PER_NODE=N` | match your nodes' GPU count |
| `--mem` | `460800` (≈ 450 GB) | `MEM=MB` | match your node memory |

Recommended: do a one-line dry-run sanity-check on the headers:

```bash
DRY_RUN=1 ACCOUNT=your_account PARTITION=your_partition \
  bash cluster_scripts/submit_all.sh q_cb_pra
cat cluster_scripts/generated/q_cb_pra.slurm | head -15
```

That prints the sbatch file with your overrides applied. Inspect the SBATCH
headers; if anything's wrong for your cluster, fix and re-run.

---

## 7. Smoke test on ONE cell first

**Always do this before submitting all 20.** It catches 90% of failure modes
in 1 hour at the cost of one node-hour:

```bash
# Load your env into the submitting shell
source .env

# Submit just q_cb_pra (the cheapest trained cell — non-LAT, Qwen)
ACCOUNT=your_account PARTITION=your_partition \
  bash cluster_scripts/submit_all.sh q_cb_pra
```

Expected timeline:
- t = 0 min: sbatch queued.
- t = 2–10 min: node allocated, model download starts (Qwen3-8B, ~16 GB).
- t = ~25 min: training step 0 begins on GPU 0.
- t = ~55 min: training finishes; `READY` written; **adapter pushed to HF**.
- t = ~55 min onward: 7 attacks + 2 benign evals run on GPUs 0–3 in parallel
  (priority order: soft_prompt_100 → direct_300 → ...). Each attack pushes
  to HF as it finishes.
- t = ~80 min: cell complete.

Watch from a separate shell:
```bash
# Tail the slurm log
ls cluster_scripts/logs/q_cb_pra.*.out | head -1 | xargs tail -f

# Or hit the dashboard:
source .env && python cluster_scripts/sync_status.py --exp headline_rerun_full
```

Verify on HF when it's done:
- Visit `https://huggingface.co/datasets/YOUR_USER_OR_ORG/headline-rerun/tree/main/adapters`
- Should see a `q_cb_pra/` folder with `adapter_config.json`,
  `adapter_model.safetensors`, etc.
- And `attack_results/q_cb_pra/{soft_prompt_100,direct_300,...}/` populated.

If the smoke cell completes cleanly and HF has its adapter + attack results,
you're ready for the full sweep.

---

## 8. Submit the full sweep

```bash
source .env
ACCOUNT=your_account PARTITION=your_partition \
  bash cluster_scripts/submit_all.sh
```

Submits 20 sbatch jobs, one per cell. Slurm runs the first 16 in parallel on
your 16 nodes; the last 4 queue and start as soon as a node frees up.
Wall-clock from first sbatch to last cell complete: **~3 hours**.

Recommended: submit LAT cells first so they get scheduled to nodes ahead of
the cheaper cells (LAT cells take ~3 h end-to-end vs. ~80 min for the rest):

```bash
source .env
ACCOUNT=your_account PARTITION=your_partition \
  bash cluster_scripts/submit_all.sh \
    l_lat_pra l_lat_no_pra q_lat_pra q_lat_no_pra \
    l_cb_pra l_cb_no_pra l_triplet_pra l_triplet_no_pra \
    l_ce_in_pra l_ce_in_no_pra \
    q_cb_pra q_cb_no_pra q_triplet_pra q_triplet_no_pra \
    q_ce_in_pra q_ce_in_no_pra \
    ref_grayswan_rr ref_simko_triplet ref_lat_l ref_dat_l
```

---

## 9. Monitoring

```bash
# Cluster-side: who's still running?
squeue -u $USER

# Per-cell tail:
tail -f cluster_scripts/logs/<cell>.<jobid>.out
tail -f cluster_scripts/logs/<cell>.<jobid>.err

# Cross-cluster dashboard (LOCAL state + HF state):
source .env && python cluster_scripts/sync_status.py --exp headline_rerun_full

# HF state only (run this from your laptop too if you want):
HF_REPO=YOUR_USER_OR_ORG/headline-rerun HF_TOKEN=hf_... \
  python cluster_scripts/sync_status.py --hf-only --exp headline_rerun_full
```

The dashboard prints a per-cell row showing:
- `train` ✓ / · — adapter trained?
- `#attacks` 0..7 — how many attack stages have READY (locally) or are pushed (HF)
- `#evals` 0..2 — gsm8k_200 + mmlu_full

When every cell shows `train ✓ #attacks 7 #evals 2` on the HF side, you're done.

---

## 10. Troubleshooting

**A node dies mid-cell.**
The orchestrator's fingerprint+READY check is the recovery mechanism. Just
re-submit the same cell:
```bash
bash cluster_scripts/submit_all.sh <cell_name>
```
- If train was complete and pushed: it'll be skipped on restart.
- If a specific attack was complete: also skipped.
- Only the in-flight + queued stages get re-run.

**A cell fails repeatedly with the same error.**
Look at `cluster_scripts/logs/<cell>.<jobid>.err`. Common causes:
- OOM during attack: lower `batch_size` for that attack via
  `attacks.<name>.attack_config.batch_size` in the YAML.
- HF auth: `HF_TOKEN` not exported. Re-source `.env`, re-submit.
- Dataset gating: 403 on WildGuardMix → step 3.1 wasn't done.

**HF push 401/403 errors in `slurm.err`.**
Either `HF_TOKEN` doesn't have write access, or `HF_REPO` doesn't exist /
isn't owned by you. Verify:
```bash
source .env
python -c "from huggingface_hub import HfApi; print(HfApi(token='$HF_TOKEN').whoami())"
python -c "from huggingface_hub import HfApi; print(HfApi(token='$HF_TOKEN').dataset_info('$HF_REPO'))"
```
Both should succeed.

**Out-of-disk on the node.**
HF model cache is at `~/.cache/huggingface` by default. Each Llama/Qwen 8B is
~16 GB. With 4 reference models + 2 training backbones, you'll cache ~100 GB.
On a shared filesystem set `HF_HOME=/workspace/hf_cache` in `.env` to point
the cache somewhere with space. Restart cells after changing.

**You want to wipe a cell and re-train fresh.**
The fingerprint check will skip a completed cell. To force re-train:
```bash
rm -rf runs/experiments/headline_rerun_full/<cell>
bash cluster_scripts/submit_all.sh <cell>
```

**Probing failed because Why-Probe-Fails isn't on the node.**
Probing is **off by default** in `pipelines:` for that reason. If you want
to enable it: clone `https://github.com/<wpf-repo>` to the path in `WPF_ROOT`
on every node, then add a probe step to each pipeline in
`experiments/configs/headline_rerun_full.yaml`:
```yaml
pipelines:
  q_cb_pra:
    - { stage: train, defense: q_cb_pra }
    - stage: attack
      defense: q_cb_pra
      attacks: [soft_prompt_100, ...]
    - { stage: benign_eval, defense: q_cb_pra, benign_eval: gsm8k_200 }
    - { stage: benign_eval, defense: q_cb_pra, benign_eval: mmlu_full }
    - { stage: probe, defense: q_cb_pra, probe: para_hb }   # <-- add this
```

---

## 11. After everything finishes

The headline outputs you care about are all under
`https://huggingface.co/datasets/YOUR_USER_OR_ORG/headline-rerun/tree/main`:

```
adapters/<cell>/                  # 16 LoRA adapters (trained cells only)
attack_results/<cell>/<attack>/   # 20 cells × 7 attacks = 140 result dirs
                                  # each has run.json with per-step scores
eval_results/<cell>/<eval>/       # 20 cells × 2 evals = 40 result dirs
manifest/<cell>.json              # append-only completion log per cell
```

Pull them all to your laptop:
```bash
huggingface-cli download YOUR_USER_OR_ORG/headline-rerun \
    --repo-type dataset --local-dir ./headline-rerun-results
```

The downstream scripts in `scripts/plot_*.py` read directly from `run.json`
in the attack stage_dirs. To produce the headline plots, point them at
`./headline-rerun-results/attack_results/<cell>/<attack>/run.json`.

---

## TL;DR — minimum viable launch

```bash
# One-time setup
git clone <repo> /workspace/AdversariaLLM && cd /workspace/AdversariaLLM
git submodule update --init --recursive
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps && pip install lm-eval
# (accept WildGuardMix and WildJailbreak licenses on HF in your browser)
huggingface-cli login                                     # paste write token
python -c "from huggingface_hub import HfApi; HfApi().create_repo('YOU/headline-rerun', repo_type='dataset', private=True, exist_ok=True)"
python scripts/build_wildjailbreak_data.py
echo 'export HF_TOKEN=hf_...' > .env
echo 'export HF_REPO=YOU/headline-rerun' >> .env

# Smoke test
source .env && ACCOUNT=acct PARTITION=part bash cluster_scripts/submit_all.sh q_cb_pra
# (wait ~80 min, verify on HF)

# Full sweep
source .env && ACCOUNT=acct PARTITION=part bash cluster_scripts/submit_all.sh

# Watch
source .env && python cluster_scripts/sync_status.py --exp headline_rerun_full
```

If you only have time to read one section, read **§3 (HF gated datasets)** —
that's the single most common bricked-launch cause.
