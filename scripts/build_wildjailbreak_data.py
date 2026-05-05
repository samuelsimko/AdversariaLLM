"""Build WildJailbreak-only training data files for PRA ablation.

Outputs:
  data/wildjailbreak_harmful.json   — list of {prompt, output} for the harm regularizer.
                                      Combines vanilla_harmful + adversarial_harmful (where
                                      adversarial != "") rows; for adv_harmful rows we add
                                      BOTH the adversarial and vanilla variants as separate
                                      prompts (so harm CE sees both views).
  data/wildjailbreak_benign.jsonl   — list of {prompt, response} for benign KL/CE
                                      (replaces UltraChat). Same expansion logic.
  data/wildjailbreak_pairs.jsonl    — paired rows (adversarial_harmful + adversarial_benign)
                                      with the WJ schema {vanilla, adversarial, completion,
                                      data_type}. Loaded by jepa_ce.py with
                                      pair_format=wildjailbreak.

Schema (from allenai/wildjailbreak, train config, delimiter=tsv):
  - vanilla: base prompt (always present)
  - adversarial: jailbreak rewrite (only present for adversarial_* rows)
  - completion: model's response
  - data_type: one of vanilla_harmful / vanilla_benign / adversarial_harmful / adversarial_benign
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("data")
HF_NAME = "allenai/wildjailbreak"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Loading {HF_NAME} ...")
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, "train", delimiter="\t", keep_default_na=False)["train"]
    print(f"  total rows: {len(ds)}")

    harmful_records = []
    benign_records = []
    pair_records = []

    for r in ds:
        dt = r["data_type"]
        vanilla = (r["vanilla"] or "").strip()
        adversarial = (r["adversarial"] or "").strip()
        completion = (r["completion"] or "").strip()
        if not completion:
            continue

        if dt == "vanilla_harmful" and vanilla:
            harmful_records.append({"prompt": vanilla, "output": completion})
        elif dt == "adversarial_harmful" and vanilla and adversarial:
            harmful_records.append({"prompt": adversarial, "output": completion})
            harmful_records.append({"prompt": vanilla,     "output": completion})
            pair_records.append({
                "vanilla": vanilla, "adversarial": adversarial,
                "completion": completion, "data_type": dt,
            })
        elif dt == "vanilla_benign" and vanilla:
            benign_records.append({"prompt": vanilla, "response": completion})
        elif dt == "adversarial_benign" and vanilla and adversarial:
            benign_records.append({"prompt": adversarial, "response": completion})
            benign_records.append({"prompt": vanilla,     "response": completion})
            pair_records.append({
                "vanilla": vanilla, "adversarial": adversarial,
                "completion": completion, "data_type": dt,
            })

    print(f"  harmful prompts (for harm regularizer): {len(harmful_records)}")
    print(f"  benign  prompts (for KL preservation):  {len(benign_records)}")
    print(f"  pair rows (for PRA JEPA loss):          {len(pair_records)}")

    h_path = OUT_DIR / "wildjailbreak_harmful.json"
    with open(h_path, "w") as f:
        json.dump(harmful_records, f)
    print(f"  wrote {h_path}")

    b_path = OUT_DIR / "wildjailbreak_benign.jsonl"
    with open(b_path, "w") as f:
        for rec in benign_records:
            f.write(json.dumps(rec) + "\n")
    print(f"  wrote {b_path}")

    p_path = OUT_DIR / "wildjailbreak_pairs.jsonl"
    with open(p_path, "w") as f:
        for rec in pair_records:
            f.write(json.dumps(rec) + "\n")
    print(f"  wrote {p_path}")


if __name__ == "__main__":
    main()
