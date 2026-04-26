from pathlib import Path

import json

from experiments.run_experiment import (
    build_attack_command,
    build_completed_stage_index,
    load_experiment_config,
    reuse_completed_stage,
)


def test_load_experiment_config_supports_yaml(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("meta:\n  experiment_name: demo\n  output_root: runs/experiments\n", encoding="utf-8")

    cfg = load_experiment_config(config_path)

    assert cfg["meta"]["experiment_name"] == "demo"


def test_build_attack_command_includes_peft_and_overrides(tmp_path):
    experiment_cfg = {
        "default_attack_dataset": "adv_behaviors",
        "classifiers": ["strong_reject"],
        "models": {
            "llama3": {"from_registry": "meta-llama/Meta-Llama-3-8B-Instruct"},
        },
    }
    defense_cfg = {
        "script": "defenses/honeypot_cb.py",
        "base_model": "llama3",
        "output_subdir": "cb_llama3",
    }
    attack_cfg = {
        "attack": "gcg",
        "attack_overrides": {"num_steps": 42, "search_width": 64},
    }
    registry_models = {
        "meta-llama/Meta-Llama-3-8B-Instruct": {
            "id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "short_name": "Llama",
            "developer_name": "Meta",
            "compile": False,
            "dtype": "bfloat16",
            "chat_template": "llama-3-instruct",
            "trust_remote_code": True,
        }
    }
    registry_datasets = {"adv_behaviors": {"name": "adv_behaviors"}}
    registry_attacks = {"gcg": {"name": "gcg"}}
    stage_dir = tmp_path / "attack_stage"

    command, fingerprint = build_attack_command(
        experiment_cfg=experiment_cfg,
        pipeline_name="demo_pipeline",
        defense_name="cb_llama3",
        defense_cfg=defense_cfg,
        attack_name="gcg_fast",
        attack_cfg=attack_cfg,
        out_root=tmp_path,
        stage_dir=stage_dir,
        python_bin="/usr/bin/python3",
        registry_models=registry_models,
        registry_datasets=registry_datasets,
        registry_attacks=registry_attacks,
        hydra_launcher="basic",
    )

    joined = " ".join(command)
    assert "model=meta-llama/Meta-Llama-3-8B-Instruct" in joined
    assert "dataset=adv_behaviors" in joined
    assert "attack=gcg" in joined
    assert "+model_overrides.peft_path=" in joined
    assert "attacks.gcg.num_steps=42" in joined
    assert fingerprint["resolved_attack"]["name"] == "gcg"
    assert fingerprint["classifiers"] == ["strong_reject"]


def test_reuse_completed_stage_copies_matching_ready_stage(tmp_path):
    source = tmp_path / "source_stage"
    source.mkdir()
    fingerprint = {"stage": "attack", "attack": "direct", "model": "demo"}
    (source / "fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    (source / "READY").write_text("", encoding="utf-8")
    (source / "artifact.txt").write_text("done", encoding="utf-8")

    index = build_completed_stage_index(tmp_path)
    target = tmp_path / "target_stage"

    reused_from = reuse_completed_stage(
        stage_dir=target,
        fingerprint=fingerprint,
        completed_stage_index=index,
    )

    assert reused_from == source
    assert (target / "READY").exists()
    assert (target / "artifact.txt").read_text(encoding="utf-8") == "done"
    reused_meta = json.loads((target / "REUSED_FROM.json").read_text(encoding="utf-8"))
    assert reused_meta["reused_from"] == str(source)
