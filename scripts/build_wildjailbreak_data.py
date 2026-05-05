"""Build aligned WildJailbreak ↔ WildGuardMix training data for PRA (no confounder).

The naive approach was to use WildGuardMix prompts/responses for harm CE and WildJailbreak
pairs for the JEPA loss. But those are DIFFERENT prompt distributions — the harm CE was
seeing prompts that pair JEPA never saw, and vice versa. That's a distribution confounder.

Fix: only use rows where the WJ pair (vanilla, adversarial) has at least one prompt
that ALSO appears in WildGuardMix's harmful-response (or compliance-response) map.
That gives us aligned (vanilla, adversarial, real_harmful_response) triples — same
prompt distribution for both losses.

Output:
  data/wildjailbreak_harmful.json   — list of {prompt, output} where prompt is one
                                      of the WJ vanilla/adversarial harmful prompts
                                      AND has a real harmful response in WildGuardMix.
                                      ~5k entries.
  data/wildjailbreak_benign.jsonl   — list of {prompt, response} where prompt is one
                                      of the WJ vanilla/adversarial benign prompts
                                      AND has a real compliance response in WildGuardMix.
                                      ~9k entries.
  data/wildjailbreak_pairs.jsonl    — ALL WJ adversarial_harmful + adversarial_benign
                                      pairs (~161k). The completion field is the WJ
                                      original (safe refusal for harmful, helpful for
                                      benign). With include_pair_harmful_in_harm_ce=false
                                      in the training config, the harmful pair completion
                                      is unused (only prompt-pair JEPA matters).

Filters on WG responses: drop chat-format-tag prefixes ([ASS], [USER], etc.) and
responses < 200 chars (often degenerate "I'd be happy to help!").
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path("data")
WJ_NAME = "allenai/wildjailbreak"
WG_NAME = "allenai/wildguardmix"
BAD_PREFIXES = ("[", "USER:", "ASSISTANT:", "Assistant:", "User:")
MIN_RESP_LEN = 200
N_PER_POOL = 5000  # cap harmful and benign pools at this size each (balanced)
SEED = 42


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    from datasets import load_dataset

    print(f"Loading {WG_NAME} (wildguardtrain) ...")
    wg = load_dataset(WG_NAME, "wildguardtrain")["train"]
    print(f"  total rows: {len(wg)}")

    # Build clean (prompt → response) maps from WildGuardMix
    wg_harm = {}
    wg_compl = {}
    for r in wg:
        prompt = (r.get("prompt") or "").strip()
        resp = (r.get("response") or "").strip()
        if not (prompt and resp):
            continue
        if any(resp.startswith(p) for p in BAD_PREFIXES) or len(resp) < MIN_RESP_LEN:
            continue
        if r.get("prompt_harm_label") == "harmful" and r.get("response_harm_label") == "harmful":
            wg_harm.setdefault(prompt, resp)
        elif (r.get("prompt_harm_label") == "unharmful"
                and r.get("response_refusal_label") == "compliance"):
            wg_compl.setdefault(prompt, resp)
    print(f"  WG clean harmful map: {len(wg_harm)} entries")
    print(f"  WG clean compliance map: {len(wg_compl)} entries")

    print(f"\nLoading {WJ_NAME} (train) ...")
    wj = load_dataset(WJ_NAME, "train", delimiter="\t", keep_default_na=False)["train"]
    print(f"  total rows: {len(wj)}")

    harmful_records = []  # list of {prompt, output}
    benign_records = []   # list of {prompt, response}
    pair_records = []     # list of {vanilla, adversarial, completion, data_type}

    n_harm_aligned_van = n_harm_aligned_adv = 0
    n_benign_aligned_van = n_benign_aligned_adv = 0

    for r in wj:
        dt = r["data_type"]
        vanilla = (r["vanilla"] or "").strip()
        adversarial = (r["adversarial"] or "").strip()
        completion = (r["completion"] or "").strip()
        if not (vanilla and adversarial and completion):
            continue

        if dt == "adversarial_harmful":
            van_resp = wg_harm.get(vanilla)
            adv_resp = wg_harm.get(adversarial)
            if van_resp:
                harmful_records.append({"prompt": vanilla, "output": van_resp})
                n_harm_aligned_van += 1
            if adv_resp:
                harmful_records.append({"prompt": adversarial, "output": adv_resp})
                n_harm_aligned_adv += 1
            # Only include adversarial_harmful pair in the pair file if at least one
            # of its prompts has an aligned real harmful response. For the completion
            # field, prefer the adversarial-aligned response (matches the adv prompt's
            # behaviour); fall back to the vanilla-aligned response.
            aligned_resp = adv_resp or van_resp
            if aligned_resp is not None:
                pair_records.append({
                    "vanilla": vanilla, "adversarial": adversarial,
                    "completion": aligned_resp, "data_type": dt,
                })

        elif dt == "adversarial_benign":
            van_resp = wg_compl.get(vanilla)
            adv_resp = wg_compl.get(adversarial)
            if van_resp:
                benign_records.append({"prompt": vanilla, "response": van_resp})
                n_benign_aligned_van += 1
            if adv_resp:
                benign_records.append({"prompt": adversarial, "response": adv_resp})
                n_benign_aligned_adv += 1
            pair_records.append({
                "vanilla": vanilla, "adversarial": adversarial,
                "completion": completion, "data_type": dt,
            })

    print(f"\n  Aligned harmful (prompt, real harmful response):")
    print(f"    via WJ vanilla ∈ WG-harmful: {n_harm_aligned_van}")
    print(f"    via WJ adversarial ∈ WG-harmful: {n_harm_aligned_adv}")
    print(f"    total entries before cap: {len(harmful_records)}")

    print(f"\n  Aligned benign (prompt, real compliance response):")
    print(f"    via WJ vanilla ∈ WG-compl: {n_benign_aligned_van}")
    print(f"    via WJ adversarial ∈ WG-compl: {n_benign_aligned_adv}")
    print(f"    total entries before cap: {len(benign_records)}")

    print(f"\n  WJ pair rows before cap: harmful={sum(1 for r in pair_records if r['data_type']=='adversarial_harmful')}, benign={sum(1 for r in pair_records if r['data_type']=='adversarial_benign')}")

    # Cap each pool at N_PER_POOL with deterministic shuffle
    rng = random.Random(SEED)
    rng.shuffle(harmful_records)
    rng.shuffle(benign_records)
    harmful_records = harmful_records[:N_PER_POOL]
    benign_records = benign_records[:N_PER_POOL]

    # Cap pair pools separately (harmful and benign each capped at N_PER_POOL)
    harmful_pairs = [r for r in pair_records if r["data_type"] == "adversarial_harmful"]
    benign_pairs  = [r for r in pair_records if r["data_type"] == "adversarial_benign"]
    rng.shuffle(harmful_pairs)
    rng.shuffle(benign_pairs)
    harmful_pairs = harmful_pairs[:N_PER_POOL]
    benign_pairs  = benign_pairs[:N_PER_POOL]
    pair_records = harmful_pairs + benign_pairs
    rng.shuffle(pair_records)

    print(f"\n  After capping at {N_PER_POOL} each:")
    print(f"    wildjailbreak_harmful.json: {len(harmful_records)}")
    print(f"    wildjailbreak_benign.jsonl: {len(benign_records)}")
    print(f"    wildjailbreak_pairs.jsonl: {len(pair_records)} ({len(harmful_pairs)} harmful + {len(benign_pairs)} benign)")

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
