# Judges

This repo now supports two judge backends in the existing judging pipeline:

- `judgezoo` for the original library behavior
- `local` for the custom judges defined in [judges.py](/workspace/AdversariaLLM/judges.py)

The integration is wired into both [run_judges.py](/workspace/AdversariaLLM/run_judges.py) and [run_sampling.py](/workspace/AdversariaLLM/run_sampling.py).

## How Selection Works

Judge selection is encoded directly in the `classifier` string.

- Plain names use `judgezoo`
- `local:<name>` uses the local backend

Examples:

```bash
python run_judges.py classifier=harmbench
python run_judges.py classifier=strong_reject
python run_judges.py classifier=local:ensemble
python run_judges.py classifier=local:harmbench
python run_judges.py classifier=local:strongreject
python run_judges.py classifier=local:wildguard
python run_judges.py classifier=local:jailjudge
python run_judges.py classifier=local:gpt_oss
```

You can also use the same strings in `conf/config.yaml` under `classifiers` when running [run_attacks.py](/workspace/AdversariaLLM/run_attacks.py).

## Stored Score Keys

For `judgezoo`, the stored score key remains the classifier name, for example:

- `harmbench`
- `strong_reject`

For local judges, the stored score key is backend-qualified, for example:

- `local:ensemble`
- `local:harmbench`

This matters because [run_sampling.py](/workspace/AdversariaLLM/run_sampling.py) reuses the score keys found in previous runs to decide which judges to rerun.

## Dual-Context Scoring

Local judges now score each completion in two contexts:

- `score_without_jailbreak`: score using the original harmful target prompt alone
- `score_with_jailbreak`: score using the attacked prompt actually sent at that step

This is intentional. In many attacks, the attacked prompt can differ a lot from the original target prompt, so storing only one score can hide false positives or context mismatches.

For `local:ensemble`, the pipeline also stores the normalized component scores for both contexts when available, for example:

- `harmbench_without_jailbreak`
- `harmbench_with_jailbreak`
- `wildguard_without_jailbreak`
- `wildguard_with_jailbreak`
- `gpt_oss_without_jailbreak`
- `gpt_oss_with_jailbreak`
- `ensemble_mean_without_jailbreak`
- `ensemble_mean_with_jailbreak`

## Local Judge Set

The local backend currently supports these names:

- `local:ensemble`
- `local:harmbench`
- `local:strongreject`
- `local:wildguard`
- `local:jailjudge`
- `local:gpt_oss`

The local path uses [judges.py](/workspace/AdversariaLLM/judges.py) and defaults to not using the HarmBench honeypot LoRA branch.

## Notes

- `judgezoo` remains the default behavior for backward compatibility.
- The vendored `strong_reject` submodule is added to `sys.path` automatically by [judges.py](/workspace/AdversariaLLM/judges.py).
- Existing files scored with plain `harmbench` or other original names will keep using the original backend unless you explicitly score them with a `local:*` classifier too.
