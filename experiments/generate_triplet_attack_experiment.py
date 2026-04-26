#!/usr/bin/env python3

from pathlib import Path

import yaml


OUT_PATH = Path("experiments/configs/triplet_attack_suite.yaml")


EXPERIMENT = {
    "meta": {
        "experiment_name": "triplet_attack_suite",
        "output_root": "runs/experiments",
    },
    "runtime": {
        "env_file": ".env",
    },
    "cluster": {
        "partition": "tamper_resistance",
        "account": "replace_me",
        "gres": "gpu:1",
        "time_attack": "03:00:00",
    },
    "classifiers": ["strong_reject", "local:ensemble"],
    "models": {
        "llama3_triplet": {
            "from_registry": "samuelsimko/Meta-Llama-3-8B-Instruct-Triplet",
        },
        "llama3_rr": {
            "from_registry": "GraySwanAI/Llama-3-8B-Instruct-RR",
        },
    },
    "datasets": {
        "jbb_inpainting_subset": {
            "name": "jbb_behaviors",
            "seed": 0,
            "idx": list(range(10)),
            "shuffle": True,
        },
        "adv_behaviors_subset": {
            "name": "adv_behaviors",
            "seed": 0,
            "idx": list(range(50)),
            "shuffle": True,
        },
    },
    "defenses": {
        "triplet_llama3": {
            "script": None,
            "base_model": "llama3_triplet",
            "output_subdir": "triplet_llama3",
        },
        "rr_llama3": {
            "script": None,
            "base_model": "llama3_rr",
            "output_subdir": "rr_llama3",
        },
    },
    "attacks": {
        "inpainting_first": {
            "attack": "inpainting",
            "dataset": "jbb_inpainting_subset",
            "attack_overrides": {
                "num_samples_per_behavior": 64,
            },
            "classifiers": ["strong_reject"],
        },
        "direct_second": {
            "attack": "direct",
            "dataset": "adv_behaviors_subset",
        },
        "prefilling_third": {
            "attack": "prefilling",
            "dataset": "adv_behaviors_subset",
            "attack_overrides": {
                "batch_size": 64,
            },
        },
        "best_of_n_fourth": {
            "attack": "bon",
            "dataset": "adv_behaviors_subset",
            "attack_overrides": {
                "num_steps": 200,
                "sigma": 0.4,
            },
        },
    },
    "pipelines": {
        "triplet_llama3": [
            {
                "stage": "attack",
                "defense": "triplet_llama3",
                "attacks": [
                    "inpainting_first",
                    "direct_second",
                    "prefilling_third",
                    "best_of_n_fourth",
                ],
            },
        ],
        "rr_llama3": [
            {
                "stage": "attack",
                "defense": "rr_llama3",
                "attacks": [
                    "inpainting_first",
                    "direct_second",
                    "prefilling_third",
                    "best_of_n_fourth",
                ],
            },
        ],
    },
}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(yaml.safe_dump(EXPERIMENT, sort_keys=False), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
