from types import SimpleNamespace

from omegaconf import OmegaConf

from run_attacks import collect_configs


class _StubDataset:
    config_idx = [0, 1, 2]

    def __len__(self):
        return 3


def test_collect_configs_merges_model_dataset_and_attack_overrides(monkeypatch):
    monkeypatch.setattr(
        "run_attacks.PromptDataset.from_name",
        lambda _name: (lambda _params: _StubDataset()),
    )
    monkeypatch.setattr("run_attacks.filter_config", lambda run_config, dset_len, overwrite=False: run_config)

    cfg = OmegaConf.create(
        {
            "model": "demo_model",
            "dataset": "demo_dataset",
            "attack": "demo_attack",
            "overwrite": False,
            "models": {
                "demo_model": {
                    "id": "base-model",
                    "tokenizer_id": "base-model",
                    "short_name": "Base",
                    "developer_name": "Demo",
                    "compile": False,
                    "dtype": "bfloat16",
                    "chat_template": None,
                    "trust_remote_code": True,
                }
            },
            "datasets": {
                "demo_dataset": {
                    "name": "demo_dataset",
                    "seed": 0,
                    "idx": None,
                }
            },
            "attacks": {
                "demo_attack": {
                    "name": "direct",
                    "num_steps": 10,
                }
            },
            "model_overrides": {"peft_path": "/tmp/adapter"},
            "dataset_overrides": {"seed": 7},
            "attack_overrides": {"num_steps": 25},
        }
    )

    run_configs = collect_configs(cfg)

    assert len(run_configs) == 1
    run_config = run_configs[0]
    assert run_config.model_params["peft_path"] == "/tmp/adapter"
    assert run_config.dataset_params["seed"] == 7
    assert run_config.attack_params["num_steps"] == 25
