"""Generate the ablation cell list (Phase 1, 60 cells).

Phase 1 is a one-axis sweep around an anchor:
  anchor = (w_jepa=5, predLR=1, layer=30, depth=2, K=256)

For each (harm_reg, backbone) we emit:
  - 1 anchor PRA cell (w_jepa=5, anchor for the rest)
  - 1 matched no-PRA control (w_jepa=0, anchor for the rest)
  - 3 w_jepa variations: {1, 10, 20}     (anchor for the rest)
  - 1 predLR variation: 100              (anchor for the rest)
  - 2 layer variations: {20, 25}         (anchor for the rest)
  - 1 depth variation: 3                 (anchor for the rest)
  - 2 K variations: {64, 512}            (anchor for the rest)
That's 1 + 1 + 3 + 1 + 2 + 1 + 2 = 11 cells per (harm_reg, backbone).

6 (harm_reg × backbone) × 11 = 66 cells.

Writes a JSON list to --out (default: scripts/ablation_cells.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HARM_REGS = ["circuit_breaker", "ce_floor", "triplet"]
BACKBONES = {
    "l": "meta-llama/Meta-Llama-3-8B-Instruct",
    "q": "Qwen/Qwen3-8B",
}

# Anchor knob values shared across all (harm_reg, backbone) cells.
ANCHOR = dict(
    w_jepa=5.0,
    predictor_lr_multiplier=1.0,
    align_layer=30,
    predictor_type="mlp",
    predictor_layers=2,
    predictor_bottleneck_dim=256,
)


def short_hr(hr: str) -> str:
    return {"circuit_breaker": "cb", "ce_floor": "ceflr", "triplet": "tri"}[hr]


def make_cell(
    backbone: str,
    harm_reg: str,
    overrides: dict[str, Any],
    suffix: str,
) -> dict[str, Any]:
    args = dict(ANCHOR)
    args.update(overrides)
    if args["predictor_type"] == "identity":
        pred_tag = "Pid"
    elif args["predictor_type"] == "linear":
        pred_tag = "Plin"
    else:
        pred_tag = f"d{args['predictor_layers']}_K{args['predictor_bottleneck_dim']}"
    name_parts = [
        backbone,
        short_hr(harm_reg),
        f"wj{args['w_jepa']:g}",
        f"plr{args['predictor_lr_multiplier']:g}",
        f"L{args['align_layer']}",
        pred_tag,
        suffix,
    ]
    name = "_".join(p for p in name_parts if p)
    return dict(
        name=name,
        backbone=backbone,
        harm_regularizer=harm_reg,
        **args,
    )


def cells_for(backbone: str, harm_reg: str) -> list[dict[str, Any]]:
    out = []
    out.append(make_cell(backbone, harm_reg, dict(),                                 "anchor"))
    out.append(make_cell(backbone, harm_reg, dict(w_jepa=0.0),                       "noPRA"))
    for wj in (1.0, 10.0, 20.0):
        out.append(make_cell(backbone, harm_reg, dict(w_jepa=wj),                     ""))
    out.append(make_cell(backbone, harm_reg, dict(predictor_lr_multiplier=100.0),    ""))
    for layer in (20, 25):
        out.append(make_cell(backbone, harm_reg, dict(align_layer=layer),             ""))
    out.append(make_cell(backbone, harm_reg, dict(predictor_layers=3),               ""))
    for k in (64, 512):
        out.append(make_cell(backbone, harm_reg, dict(predictor_bottleneck_dim=k),    ""))
    # P=Identity: predictor is the identity function (no learned head). Tests
    # whether the JEPA-loss benefit comes from the learned predictor mapping
    # or just from enforcing similarity between adv and clean reps directly.
    out.append(make_cell(backbone, harm_reg, dict(predictor_type="identity"),        ""))
    # Dedup by name (anchor can equal a knob's anchor value).
    seen = set()
    deduped = []
    for c in out:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        deduped.append(c)
    return deduped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/ablation_cells.json")
    args = ap.parse_args()

    cells: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for harm_reg in HARM_REGS:
            cells.extend(cells_for(backbone, harm_reg))

    # Resolve full HF model id.
    for c in cells:
        c["base_model"] = BACKBONES[c["backbone"]]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cells, f, indent=2)
    print(f"[build_ablation_cells] wrote {len(cells)} cells -> {args.out}")
    # Sanity-print first few names.
    for c in cells[:6]:
        print(f"  {c['name']}")
    print("  ...")


if __name__ == "__main__":
    main()
