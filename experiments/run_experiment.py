#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local alias so the probe-stage builder reads cleanly.
shlex_quote = shlex.quote

import yaml

from experiments.backends.local import LocalBackend
from experiments.backends.local_gpu import LocalGPUBackend
from experiments.backends.mock import MockBackend
from experiments.backends.slurm import SlurmBackend


REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"


def stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Experiment config at {path} must be a mapping.")
        return data
    raise ValueError(f"Unsupported experiment config format: {path.suffix}")


def load_registry(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Registry config at {path} must be a mapping.")
    return data


def build_arg_list(args: Dict[str, object]) -> List[str]:
    out: List[str] = []
    for k, v in args.items():
        if v is None:
            continue
        out.append(f"--{k}")
        if isinstance(v, list):
            out.extend(str(x) for x in v)
        else:
            out.append(str(v))
    return out


def get_time(cluster: dict, key: str, default: str) -> str:
    if "time" in cluster and key in cluster["time"]:
        return cluster["time"][key]
    return cluster.get(f"time_{key}", default)


def is_stage_done(out_dir: Path, fingerprint: dict) -> bool:
    fp_path = out_dir / "fingerprint.json"
    ready = out_dir / "READY"
    if not (fp_path.exists() and ready.exists()):
        return False
    try:
        old_fp = json.loads(fp_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return old_fp == fingerprint


def build_completed_stage_index(out_root: Path) -> dict[str, list[Path]]:
    completed: dict[str, list[Path]] = {}
    if not out_root.exists():
        return completed
    for fingerprint_path in out_root.rglob("fingerprint.json"):
        stage_dir = fingerprint_path.parent
        if not (stage_dir / "READY").exists():
            continue
        try:
            fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        completed.setdefault(stable_hash(fingerprint), []).append(stage_dir)
    return completed


def copy_stage_contents(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def reuse_completed_stage(
    *,
    stage_dir: Path,
    fingerprint: dict[str, Any],
    completed_stage_index: dict[str, list[Path]],
) -> Path | None:
    fingerprint_key = stable_hash(fingerprint)
    for candidate in completed_stage_index.get(fingerprint_key, []):
        if candidate.resolve() == stage_dir.resolve():
            continue
        if not is_stage_done(candidate, fingerprint):
            continue
        copy_stage_contents(candidate, stage_dir)
        (stage_dir / "REUSED_FROM.json").write_text(
            json.dumps(
                {
                    "reused_from": str(candidate),
                    "fingerprint_hash": fingerprint_key,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if (stage_dir / "FAILED").exists():
            (stage_dir / "FAILED").unlink()
        (stage_dir / "READY").touch()
        completed_stage_index.setdefault(fingerprint_key, []).append(stage_dir)
        return candidate
    return None


_LIST_RANGE_RE = re.compile(r"^\s*list\(\s*range\(\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*)?)?\)\s*\)\s*$")


def hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        m = _LIST_RANGE_RE.match(value)
        if m:
            a, b, c = m.groups()
            args = [int(a)] + ([int(b)] if b is not None else []) + ([int(c)] if c is not None else [])
            expanded = list(range(*args))
            return "[" + ",".join(str(i) for i in expanded) + "]"
    return json.dumps(value)


def flatten_hydra_overrides(prefix: str, value: Any, *, add: bool = False) -> List[str]:
    if isinstance(value, dict):
        out: List[str] = []
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(flatten_hydra_overrides(next_prefix, nested, add=add))
        return out
    assign = "+=" if False else "="
    return [f"{'+' if add else ''}{prefix}{assign}{hydra_value(value)}"]


def append_submission_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def wrap_job_command(
    *,
    python_bin: str,
    stage_dir: Path,
    job_name: str,
    job_type: str,
    fingerprint: dict[str, Any],
    command: list[str],
    allow_failure: bool = False,
) -> list[str]:
    fingerprint_b64 = base64.b64encode(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    wrapped = [
        python_bin,
        str(REPO_ROOT / "experiments" / "execute_job.py"),
        "--stage-dir",
        str(stage_dir),
        "--job-name",
        job_name,
        "--job-type",
        job_type,
        "--fingerprint-b64",
        fingerprint_b64,
        "--cwd",
        str(REPO_ROOT),
    ]
    if allow_failure:
        wrapped.append("--allow-failure")
    wrapped.extend(["--", *command])
    return wrapped


def resolve_registry_key(spec_name: str, spec: Any, field_names: list[str]) -> Optional[str]:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        for field in field_names:
            if spec.get(field):
                return spec[field]
    if isinstance(spec_name, str) and spec_name:
        return None
    return None


def normalize_inline_model(alias: str, spec: dict[str, Any]) -> dict[str, Any]:
    data = dict(spec)
    if "hf_id" in data and "id" not in data:
        data["id"] = data.pop("hf_id")
    if "id" not in data:
        raise ValueError(f"Inline model '{alias}' must define `id` or legacy `hf_id`.")
    data.setdefault("tokenizer_id", data["id"])
    data.setdefault("short_name", alias)
    data.setdefault("developer_name", "Experiment")
    data.setdefault("compile", False)
    data.setdefault("dtype", "bfloat16")
    data.setdefault("chat_template", None)
    data.setdefault("trust_remote_code", True)
    return data


def normalize_inline_attack(alias: str, spec: dict[str, Any]) -> dict[str, Any]:
    data = dict(spec)
    if "attack_name" in data and "name" not in data:
        data["name"] = data["attack_name"]
    data.setdefault("name", alias)
    return data


def load_legacy_attack_overrides(attack_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    payload: Any
    if "attack_config" in spec:
        payload = spec["attack_config"]
    elif "attack_config_path" in spec:
        payload = load_experiment_config(spec["attack_config_path"])
    else:
        return {}

    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError(
                f"Legacy attack '{attack_name}' must define a single config object to map onto the library pipeline."
            )
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy attack '{attack_name}' produced a non-dict config payload.")
    return payload


def resolve_model_for_attack(
    *,
    defense_name: str,
    defense_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    registry_models: dict[str, Any],
    out_root: Path,
) -> tuple[str, str, list[str], dict[str, Any]]:
    model_ref = defense_cfg["base_model"]
    model_spec = models_cfg.get(model_ref, model_ref)
    synthetic_key = f"exp_model_{slugify(defense_name)}"
    extra_overrides: list[str] = []

    registry_key = resolve_registry_key(model_ref, model_spec, ["from_registry", "registry_model"])
    if registry_key:
        model_key = registry_key
        if registry_key not in registry_models:
            raise ValueError(f"Model '{registry_key}' is not present in conf/models/models.yaml")
        model_params = dict(registry_models[registry_key])
    else:
        if not isinstance(model_spec, dict):
            raise ValueError(f"Model '{model_ref}' must be a registry name or mapping.")
        model_key = synthetic_key
        model_params = normalize_inline_model(model_ref, model_spec)
        extra_overrides.extend(flatten_hydra_overrides(f"models.{model_key}", model_params, add=True))

    lora_path = None
    if defense_cfg.get("script") is not None and defense_cfg.get("attack_uses_adapter", True):
        adapter_subdir = defense_cfg.get("adapter_subdir", "lora_adapter")
        lora_path = out_root / defense_cfg["output_subdir"] / adapter_subdir
        extra_overrides.extend(flatten_hydra_overrides("model_overrides.peft_path", str(lora_path), add=True))
        model_params["peft_path"] = str(lora_path)

    return model_key, model_params["id"], extra_overrides, model_params


def resolve_dataset_for_attack(
    attack_name: str,
    attack_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    registry_datasets: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    datasets_cfg = experiment_cfg.get("datasets", {})
    dataset_ref = attack_cfg.get("dataset", experiment_cfg.get("default_attack_dataset", "adv_behaviors"))
    dataset_spec = datasets_cfg.get(dataset_ref, dataset_ref)
    synthetic_key = f"exp_dataset_{slugify(attack_name)}"
    extra_overrides: list[str] = []
    dataset_uses_existing_key = False

    registry_key = resolve_registry_key(str(dataset_ref), dataset_spec, ["from_registry", "registry_dataset"])
    if registry_key:
        if registry_key not in registry_datasets:
            raise ValueError(f"Dataset '{registry_key}' is not present in conf/datasets/datasets.yaml")
        dataset_key = registry_key
        dataset_params = dict(registry_datasets[registry_key])
        dataset_uses_existing_key = True
    else:
        if not isinstance(dataset_spec, dict):
            raise ValueError(f"Dataset '{dataset_ref}' must be a registry name or mapping.")
        dataset_params = dict(dataset_spec)
        dataset_params.setdefault("name", str(dataset_ref))
        dataset_name = str(dataset_params["name"])
        if dataset_name in registry_datasets:
            dataset_key = dataset_name
            dataset_uses_existing_key = True
        else:
            dataset_key = synthetic_key
            extra_overrides.extend(flatten_hydra_overrides(f"datasets.{dataset_key}", dataset_params, add=True))

    dataset_overrides = attack_cfg.get("dataset_overrides", {})
    if dataset_uses_existing_key:
        if not registry_key:
            extra_overrides.extend(flatten_hydra_overrides(f"datasets.{dataset_key}", dataset_params))
        extra_overrides.extend(flatten_hydra_overrides(f"datasets.{dataset_key}", dataset_overrides))
    else:
        extra_overrides.extend(flatten_hydra_overrides(f"datasets.{dataset_key}", dataset_overrides))
    return dataset_key, extra_overrides, dataset_params


def resolve_attack_runtime(
    attack_name: str,
    attack_cfg: dict[str, Any],
    registry_attacks: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    synthetic_key = f"exp_attack_{slugify(attack_name)}"
    extra_overrides: list[str] = []

    registry_key = resolve_registry_key(attack_name, attack_cfg, ["attack", "from_registry", "library_attack"])
    legacy_name = attack_cfg.get("attack_name") if isinstance(attack_cfg, dict) else None

    if registry_key:
        attack_key = registry_key
        if registry_key not in registry_attacks:
            raise ValueError(f"Attack '{registry_key}' is not present in conf/attacks/attacks.yaml")
        attack_params = dict(registry_attacks[registry_key])
    elif legacy_name:
        attack_key = legacy_name
        if attack_key not in registry_attacks:
            raise ValueError(
                f"Legacy attack '{attack_key}' is not supported by the current library pipeline. "
                "Use a library attack from conf/attacks/attacks.yaml in the experiment YAML."
            )
        attack_params = dict(registry_attacks[attack_key])
    else:
        attack_key = synthetic_key
        attack_params = normalize_inline_attack(attack_name, attack_cfg)
        extra_overrides.extend(flatten_hydra_overrides(f"attacks.{attack_key}", attack_params, add=True))

    attack_overrides = attack_cfg.get("attack_overrides", {})
    if isinstance(attack_cfg, dict) and ("attack_config" in attack_cfg or "attack_config_path" in attack_cfg):
        attack_overrides = {**load_legacy_attack_overrides(attack_name, attack_cfg), **attack_overrides}
    if registry_key or legacy_name:
        extra_overrides.extend(flatten_hydra_overrides(f"attacks.{attack_key}", attack_overrides))
    else:
        extra_overrides.extend(flatten_hydra_overrides("attack_overrides", attack_overrides, add=True))
    return attack_key, extra_overrides, attack_params


def build_attack_command(
    *,
    experiment_cfg: dict[str, Any],
    pipeline_name: str,
    defense_name: str,
    defense_cfg: dict[str, Any],
    attack_name: str,
    attack_cfg: dict[str, Any],
    out_root: Path,
    stage_dir: Path,
    python_bin: str,
    registry_models: dict[str, Any],
    registry_datasets: dict[str, Any],
    registry_attacks: dict[str, Any],
    hydra_launcher: str,
) -> tuple[list[str], dict[str, Any]]:
    model_key, model_id, model_overrides, model_params = resolve_model_for_attack(
        defense_name=defense_name,
        defense_cfg=defense_cfg,
        models_cfg=experiment_cfg.get("models", {}),
        registry_models=registry_models,
        out_root=out_root,
    )
    dataset_key, dataset_overrides, dataset_params = resolve_dataset_for_attack(
        attack_name, attack_cfg, experiment_cfg, registry_datasets
    )
    attack_key, attack_overrides, attack_params = resolve_attack_runtime(attack_name, attack_cfg, registry_attacks)

    classifiers = attack_cfg.get(
        "classifiers",
        experiment_cfg.get("classifiers", ["strong_reject"]),
    )
    judge_selection = attack_cfg.get("judge_selection", experiment_cfg.get("judge_selection"))
    overwrite = attack_cfg.get("overwrite", experiment_cfg.get("overwrite", False))

    run_name = f"{pipeline_name}__{defense_name}__{attack_name}"
    command = [
        python_bin,
        str(REPO_ROOT / "run_attacks.py"),
        f"name={run_name}",
        f"root_dir={REPO_ROOT}",
        f"save_dir={stage_dir / 'outputs'}",
        f"embed_dir={stage_dir / 'embeddings'}",
        f"model={model_key}",
        f"dataset={dataset_key}",
        f"attack={attack_key}",
        f"overwrite={hydra_value(overwrite)}",
        f"classifiers={hydra_value(classifiers)}",
        f"hydra/launcher={hydra_launcher}",
        f"hydra.run.dir={stage_dir / 'hydra'}",
        f"hydra.sweep.dir={stage_dir / 'hydra_multirun'}",
        "hydra.output_subdir=null",
        *model_overrides,
        *dataset_overrides,
        *attack_overrides,
    ]
    if judge_selection:
        command.extend(flatten_hydra_overrides("judge_selection", judge_selection))

    fingerprint = {
        "pipeline": pipeline_name,
        "stage": "attack",
        "defense": defense_name,
        "attack": attack_name,
        "model_key": model_key,
        "model_id": model_id,
        "dataset_key": dataset_key,
        "attack_key": attack_key,
        "classifiers": classifiers,
        "judge_selection": judge_selection,
        "overwrite": overwrite,
        "resolved_model": model_params,
        "resolved_dataset": dataset_params,
        "resolved_attack": attack_params,
    }
    return command, fingerprint


def create_backend(args, cluster: dict, *, workdir: str, env_file: str | None, venv_activate: str | None):
    if args.backend == "local":
        return LocalBackend()
    if args.backend == "slurm":
        return SlurmBackend(
            partition=cluster["partition"],
            account=cluster["account"],
            gres=cluster["gres"],
            workdir=workdir,
            env_file=env_file,
            venv_activate=venv_activate,
            cpus_per_task=cluster.get("cpus_per_task"),
            mem=cluster.get("mem"),
            qos=cluster.get("qos"),
            constraint=cluster.get("constraint"),
        )
    if args.backend == "mock":
        return MockBackend()
    if args.backend == "local_gpu":
        local_gpu_params = inspect.signature(LocalGPUBackend.__init__).parameters
        if "workdir" in local_gpu_params:
            return LocalGPUBackend(
                num_gpus=cluster.get("num_gpus"),
                workdir=workdir,
                env_file=env_file,
                venv_activate=venv_activate,
            )
        return LocalGPUBackend(num_gpus=cluster.get("num_gpus"))
    raise ValueError(f"Unknown backend: {args.backend}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment YAML or legacy JSON")
    parser.add_argument("--backend", choices=["local", "slurm", "mock", "local_gpu"], default="local")
    parser.add_argument(
        "--extend-idx",
        action="store_true",
        help=(
            "Re-submit attack stages even if their stage_dir already has READY. "
            "The per-idx dedup inside run_attacks.py (filter_config + SQLite) will "
            "skip behaviors already completed, so only the *new* idx values are run. "
            "Use this to incrementally extend N=25 → N=50 → N=100 in the same "
            "stage_dir without redoing prior behaviors."
        ),
    )
    parser.add_argument(
        "--run-pipelines",
        nargs="+",
        default=None,
        help=(
            "Optional list of pipeline names to run (overrides cfg.run_pipelines). "
            "Useful for per-cell sbatch dispatch where each node only runs one cell."
        ),
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    registry_models = load_registry(CONF_DIR / "models" / "models.yaml")
    registry_datasets = load_registry(CONF_DIR / "datasets" / "datasets.yaml")
    registry_attacks = load_registry(CONF_DIR / "attacks" / "attacks.yaml")

    meta = cfg["meta"]
    cluster = cfg.get("cluster", {})
    runtime = cfg.get("runtime", {})
    continue_on_attack_failure = bool(runtime.get("continue_on_attack_failure", False))

    python_bin = runtime.get("python_bin", sys.executable)
    venv_activate = runtime.get("venv_activate")
    env_file = runtime.get("env_file")
    if env_file is not None:
        env_file = str((REPO_ROOT / env_file).resolve()) if not Path(env_file).is_absolute() else env_file

    out_root = (REPO_ROOT / meta["output_root"] / meta["experiment_name"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    backend = create_backend(
        args,
        cluster,
        workdir=str(REPO_ROOT),
        env_file=env_file,
        venv_activate=venv_activate,
    )
    hydra_launcher = runtime.get("hydra_launcher")
    if hydra_launcher is None:
        hydra_launcher = "basic" if args.backend in {"local", "local_gpu", "mock"} else "a100h100"

    models = cfg.get("models", {})
    defenses = cfg.get("defenses", {})
    attacks = cfg.get("attacks", {})
    benign_evals = cfg.get("benign_evals", {})
    probes_cfg = cfg.get("probes", {})

    pipelines = cfg.get("pipelines")
    if pipelines is None:
        pipelines = {"default": cfg["pipeline"]}
    run_pipelines = args.run_pipelines if args.run_pipelines else cfg.get("run_pipelines", list(pipelines.keys()))

    submission_manifest = {
        "experiment_name": meta["experiment_name"],
        "config_path": str(Path(args.config).resolve()),
        "backend": args.backend,
        "pipelines": run_pipelines,
    }
    (out_root / "submission_manifest.json").write_text(
        json.dumps(submission_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    jobs_path = out_root / "jobs.jsonl"
    completed_stage_index = build_completed_stage_index(out_root)

    train_jobs: Dict[str, Optional[str]] = {}

    for pipeline_name, pipeline in pipelines.items():
        if pipeline_name not in run_pipelines:
            continue
        for step in pipeline:
            if step["stage"] != "train":
                continue
            defense_name = step["defense"]
            if defense_name in train_jobs:
                continue

            ddef = defenses[defense_name]
            if ddef.get("script") is None:
                train_jobs[defense_name] = None
                continue

            model_key, model_id, _, _ = resolve_model_for_attack(
                defense_name=defense_name,
                defense_cfg=ddef,
                models_cfg=models,
                registry_models=registry_models,
                out_root=out_root,
            )
            out_dir = out_root / ddef["output_subdir"]
            train_cmd = [
                python_bin,
                str(REPO_ROOT / ddef["script"]),
                "--model",
                model_id,
                "--output_dir",
                str(out_dir),
                *build_arg_list(ddef.get("train_args", {})),
            ]
            data = ddef.get("data", {})
            if "cb_path" in data:
                train_cmd += ["--cb_path", data["cb_path"]]
            if "honeypot_path" in data:
                train_cmd += ["--honeypot_path", data["honeypot_path"]]

            fingerprint = {
                "stage": "train",
                "defense": defense_name,
                "script": ddef["script"],
                "model_key": model_key,
                "base_model": model_id,
                "train_args": ddef.get("train_args", {}),
                "data": data,
            }
            if is_stage_done(out_dir, fingerprint):
                print(f"Skipping training {defense_name} (already completed)")
                train_jobs[defense_name] = None
                continue
            reused_from = reuse_completed_stage(
                stage_dir=out_dir,
                fingerprint=fingerprint,
                completed_stage_index=completed_stage_index,
            )
            if reused_from is not None:
                print(f"Reused completed training {defense_name} from {reused_from}")
                train_jobs[defense_name] = None
                append_submission_record(
                    jobs_path,
                    {
                        "job_id": None,
                        "job_name": f"{pipeline_name}_train_{defense_name}",
                        "stage": "train",
                        "pipeline": pipeline_name,
                        "defense": defense_name,
                        "stage_dir": str(out_dir),
                        "reused_from": str(reused_from),
                        "fingerprint_hash": stable_hash(fingerprint),
                    },
                )
                continue

            job_name = f"{pipeline_name}_train_{defense_name}"
            wrapped = wrap_job_command(
                python_bin=python_bin,
                stage_dir=out_dir,
                job_name=job_name,
                job_type="train",
                fingerprint=fingerprint,
                command=train_cmd,
            )
            stdout_log = logs_dir / f"{job_name}.out"
            stderr_log = logs_dir / f"{job_name}.err"
            job_id = backend.submit(
                name=job_name,
                command=wrapped,
                time=get_time(cluster, "train", "04:00:00"),
                output_log=str(stdout_log),
                error_log=str(stderr_log),
            )
            append_submission_record(
                jobs_path,
                {
                    "job_id": job_id,
                    "job_name": job_name,
                    "stage": "train",
                    "pipeline": pipeline_name,
                    "defense": defense_name,
                    "stage_dir": str(out_dir),
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                    "fingerprint_hash": stable_hash(fingerprint),
                },
            )
            train_jobs[defense_name] = job_id

    print("Training jobs submitted.")

    # Attack-priority ordering: enqueue all attack jobs across pipelines in
    # the order specified by runtime.attack_priority (defaults to YAML order).
    # All workers pull from one FIFO queue, so submitting attacks of priority
    # 0 first means they all run before any priority-1 attack starts. We
    # collect attack stages here and submit them at the end of the main loop.
    attack_priority: list[str] = list(runtime.get("attack_priority", []))

    pending_attacks: list[dict[str, Any]] = []  # holds args for delayed attack submission

    for pipeline_name, pipeline in pipelines.items():
        if pipeline_name not in run_pipelines:
            continue
        print(f"Running pipeline: {pipeline_name}")
        for step in pipeline:
            stage = step["stage"]
            if stage == "train":
                continue

            defense_name = step["defense"]
            ddef = defenses[defense_name]
            deps: List[str] = []
            if train_jobs.get(defense_name):
                deps.append(train_jobs[defense_name])

            if stage == "attack":
                for attack_name in step["attacks"]:
                    attack_cfg = attacks[attack_name]
                    attack_stage_dir = out_root / ddef["output_subdir"] / "attacks" / pipeline_name / attack_name
                    attack_stage_dir.mkdir(parents=True, exist_ok=True)
                    command, fingerprint = build_attack_command(
                        experiment_cfg=cfg,
                        pipeline_name=pipeline_name,
                        defense_name=defense_name,
                        defense_cfg=ddef,
                        attack_name=attack_name,
                        attack_cfg=attack_cfg,
                        out_root=out_root,
                        stage_dir=attack_stage_dir,
                        python_bin=python_bin,
                        registry_models=registry_models,
                        registry_datasets=registry_datasets,
                        registry_attacks=registry_attacks,
                        hydra_launcher=hydra_launcher,
                    )
                    if is_stage_done(attack_stage_dir, fingerprint):
                        if args.extend_idx:
                            print(f"--extend-idx: re-submitting attack {attack_name} for {defense_name} (idx-level dedup will skip done behaviors)")
                            # Clear READY so the orchestrator considers the stage in-flight again.
                            ready_path = attack_stage_dir / "READY"
                            if ready_path.exists():
                                ready_path.unlink()
                        else:
                            print(f"Skipping attack {attack_name} for {defense_name} (already completed)")
                            continue
                    if not args.extend_idx:
                        reused_from = reuse_completed_stage(
                            stage_dir=attack_stage_dir,
                            fingerprint=fingerprint,
                            completed_stage_index=completed_stage_index,
                        )
                    else:
                        reused_from = None
                    if reused_from is not None:
                        print(f"Reused completed attack {attack_name} for {defense_name} from {reused_from}")
                        append_submission_record(
                            jobs_path,
                            {
                                "job_id": None,
                                "job_name": f"{pipeline_name}_attack_{attack_name}_{defense_name}",
                                "stage": "attack",
                                "pipeline": pipeline_name,
                                "defense": defense_name,
                                "attack": attack_name,
                                "stage_dir": str(attack_stage_dir),
                                "reused_from": str(reused_from),
                                "depends_on": deps,
                                "fingerprint_hash": stable_hash(fingerprint),
                            },
                        )
                        continue

                    job_name = f"{pipeline_name}_attack_{attack_name}_{defense_name}"
                    wrapped = wrap_job_command(
                        python_bin=python_bin,
                        stage_dir=attack_stage_dir,
                        job_name=job_name,
                        job_type="attack",
                        fingerprint=fingerprint,
                        command=command,
                        allow_failure=continue_on_attack_failure,
                    )
                    stdout_log = logs_dir / f"{job_name}.out"
                    stderr_log = logs_dir / f"{job_name}.err"
                    pending_attacks.append({
                        "attack_name": attack_name,
                        "pipeline_name": pipeline_name,
                        "defense_name": defense_name,
                        "job_name": job_name,
                        "wrapped": wrapped,
                        "stdout_log": stdout_log,
                        "stderr_log": stderr_log,
                        "deps": deps,
                        "stage_dir": attack_stage_dir,
                        "fingerprint": fingerprint,
                    })
            elif stage == "benign_eval":
                benign_name = step["benign_eval"]
                bcfg = benign_evals[benign_name]
                _, model_id, _, _ = resolve_model_for_attack(
                    defense_name=defense_name,
                    defense_cfg=ddef,
                    models_cfg=models,
                    registry_models=registry_models,
                    out_root=out_root,
                )
                lora_path = out_root / ddef["output_subdir"] / "lora_adapter" if ddef.get("script") is not None else None
                out_dir = out_root / ddef["output_subdir"] / "benign_eval" / pipeline_name / benign_name
                out_dir.mkdir(parents=True, exist_ok=True)

                benign_cmd = [
                    python_bin,
                    str(REPO_ROOT / "benign_capabilities" / "run_benign_eval.py"),
                    "--model",
                    model_id,
                    "--tasks",
                    bcfg["tasks"],
                    "--output-dir",
                    str(out_dir / "results"),
                ]
                if lora_path is not None:
                    benign_cmd += ["--lora", str(lora_path)]
                if "limit" in bcfg:
                    benign_cmd += ["--limit", str(bcfg["limit"])]
                if "device" in bcfg:
                    benign_cmd += ["--device", str(bcfg["device"])]
                if "dtype" in bcfg:
                    benign_cmd += ["--dtype", str(bcfg["dtype"])]

                fingerprint = {
                    "pipeline": pipeline_name,
                    "stage": "benign_eval",
                    "defense": defense_name,
                    "benign_eval": benign_name,
                    "tasks": bcfg["tasks"],
                    "limit": bcfg.get("limit"),
                    "model": model_id,
                    "lora_path": str(lora_path) if lora_path is not None else None,
                }
                if is_stage_done(out_dir, fingerprint):
                    print(f"Skipping benign eval {benign_name}")
                    continue
                reused_from = reuse_completed_stage(
                    stage_dir=out_dir,
                    fingerprint=fingerprint,
                    completed_stage_index=completed_stage_index,
                )
                if reused_from is not None:
                    print(f"Reused benign eval {benign_name} from {reused_from}")
                    append_submission_record(
                        jobs_path,
                        {
                            "job_id": None,
                            "job_name": f"{pipeline_name}_benign_{benign_name}_{defense_name}",
                            "stage": "benign_eval",
                            "pipeline": pipeline_name,
                            "defense": defense_name,
                            "benign_eval": benign_name,
                            "stage_dir": str(out_dir),
                            "reused_from": str(reused_from),
                            "depends_on": deps,
                            "fingerprint_hash": stable_hash(fingerprint),
                        },
                    )
                    continue

                job_name = f"{pipeline_name}_benign_{benign_name}_{defense_name}"
                wrapped = wrap_job_command(
                    python_bin=python_bin,
                    stage_dir=out_dir,
                    job_name=job_name,
                    job_type="benign_eval",
                    fingerprint=fingerprint,
                    command=benign_cmd,
                )
                stdout_log = logs_dir / f"{job_name}.out"
                stderr_log = logs_dir / f"{job_name}.err"
                job_id = backend.submit(
                    name=job_name,
                    command=wrapped,
                    time=get_time(cluster, "benign", "04:00:00"),
                    output_log=str(stdout_log),
                    error_log=str(stderr_log),
                    depends_on=deps,
                )
                append_submission_record(
                    jobs_path,
                    {
                        "job_id": job_id,
                        "job_name": job_name,
                        "stage": "benign_eval",
                        "pipeline": pipeline_name,
                        "defense": defense_name,
                        "benign_eval": benign_name,
                        "stage_dir": str(out_dir),
                        "stdout_log": str(stdout_log),
                        "stderr_log": str(stderr_log),
                        "depends_on": deps,
                        "fingerprint_hash": stable_hash(fingerprint),
                    },
                )
            elif stage == "probe":
                probe_name = step["probe"]
                pcfg = probes_cfg[probe_name]
                _, model_id, _, mparams = resolve_model_for_attack(
                    defense_name=defense_name,
                    defense_cfg=ddef,
                    models_cfg=models,
                    registry_models=registry_models,
                    out_root=out_root,
                )
                # resolve_model_for_attack sets model_params["peft_path"] for
                # trained-in-place defenses (out_root/<subdir>/lora_adapter); it
                # also preserves any inline-set peft_path on ref cells.
                lora_path_str = mparams.get("peft_path")
                lora_path = Path(lora_path_str) if lora_path_str else None
                out_dir = out_root / ddef["output_subdir"] / "probe" / probe_name
                out_dir.mkdir(parents=True, exist_ok=True)
                states_dir = out_dir / "states"

                wpf_root = pcfg.get("wpf_root", "/workspace/AdversariaLLM/Why-Probe-Fails")
                data_roots = list(pcfg.get("data_roots", []))
                layer_idx = int(pcfg.get("layer_idx", -1))
                probe_config_rel = pcfg.get("probe_config")
                seeds = list(pcfg.get("seeds", [42]))
                probes_list = list(pcfg.get("probes",
                                            ["svm_raw", "mlp_no_jepa", "jepa", "svm_on_jepa_z"]))
                with_analyses = bool(pcfg.get("with_analyses", False))

                if not data_roots:
                    raise ValueError(f"probe '{probe_name}': data_roots is required")
                if not probe_config_rel:
                    raise ValueError(f"probe '{probe_name}': probe_config is required")

                # Build a single bash command: chained extract_states (per data_root)
                # then a compare_probes invocation per seed. cd into wpf_root so the
                # local `jepa` package is importable without PYTHONPATH gymnastics.
                parts: list[str] = [
                    f"cd {shlex_quote(wpf_root)}",
                ]
                for dr in data_roots:
                    if isinstance(dr, str):
                        dr_path, dr_views = dr, None
                        dr_datasets = None
                    else:
                        dr_path = dr["path"]
                        dr_views = dr.get("views")
                        dr_datasets = dr.get("datasets")
                    extract = [
                        shlex_quote(python_bin),
                        "scripts/extract_states.py",
                        "--model_path", shlex_quote(model_id),
                        "--data_root", shlex_quote(dr_path),
                        "--layer_idx", str(layer_idx),
                        "--out_dir", shlex_quote(str(states_dir)),
                    ]
                    if dr_views:
                        extract += ["--views"] + [shlex_quote(v) for v in dr_views]
                    if dr_datasets:
                        extract += ["--datasets"] + [shlex_quote(d) for d in dr_datasets]
                    if lora_path is not None:
                        extract += ["--adapter_path", shlex_quote(str(lora_path))]
                    parts.append(" ".join(extract))
                for seed in seeds:
                    seed_out = out_dir / f"seed_{seed}"
                    cmp = [
                        shlex_quote(python_bin),
                        "scripts/compare_probes.py",
                        "--config", shlex_quote(probe_config_rel),
                        "--states_dir", shlex_quote(str(states_dir)),
                        "--out_dir", shlex_quote(str(seed_out)),
                        "--seed", str(seed),
                        "--probes",
                    ] + [shlex_quote(p) for p in probes_list]
                    if with_analyses:
                        cmp.append("--with_analyses")
                    parts.append(" ".join(cmp))

                probe_shell = " && ".join(parts)
                probe_cmd = ["bash", "-lc", probe_shell]

                fingerprint = {
                    "pipeline": pipeline_name,
                    "stage": "probe",
                    "defense": defense_name,
                    "probe": probe_name,
                    "model": model_id,
                    "lora_path": str(lora_path) if lora_path is not None else None,
                    "wpf_root": wpf_root,
                    "layer_idx": layer_idx,
                    "data_roots": data_roots,
                    "probe_config": probe_config_rel,
                    "seeds": seeds,
                    "probes": probes_list,
                    "with_analyses": with_analyses,
                }
                if is_stage_done(out_dir, fingerprint):
                    print(f"Skipping probe {probe_name} for {defense_name}")
                    continue
                reused_from = reuse_completed_stage(
                    stage_dir=out_dir,
                    fingerprint=fingerprint,
                    completed_stage_index=completed_stage_index,
                )
                if reused_from is not None:
                    print(f"Reused probe {probe_name} from {reused_from}")
                    append_submission_record(
                        jobs_path,
                        {
                            "job_id": None,
                            "job_name": f"{pipeline_name}_probe_{probe_name}_{defense_name}",
                            "stage": "probe",
                            "pipeline": pipeline_name,
                            "defense": defense_name,
                            "probe": probe_name,
                            "stage_dir": str(out_dir),
                            "reused_from": str(reused_from),
                            "depends_on": deps,
                            "fingerprint_hash": stable_hash(fingerprint),
                        },
                    )
                    continue

                job_name = f"{pipeline_name}_probe_{probe_name}_{defense_name}"
                wrapped = wrap_job_command(
                    python_bin=python_bin,
                    stage_dir=out_dir,
                    job_name=job_name,
                    job_type="probe",
                    fingerprint=fingerprint,
                    command=probe_cmd,
                )
                stdout_log = logs_dir / f"{job_name}.out"
                stderr_log = logs_dir / f"{job_name}.err"
                job_id = backend.submit(
                    name=job_name,
                    command=wrapped,
                    time=get_time(cluster, "probe", "04:00:00"),
                    output_log=str(stdout_log),
                    error_log=str(stderr_log),
                    depends_on=deps,
                )
                append_submission_record(
                    jobs_path,
                    {
                        "job_id": job_id,
                        "job_name": job_name,
                        "stage": "probe",
                        "pipeline": pipeline_name,
                        "defense": defense_name,
                        "probe": probe_name,
                        "stage_dir": str(out_dir),
                        "stdout_log": str(stdout_log),
                        "stderr_log": str(stderr_log),
                        "depends_on": deps,
                        "fingerprint_hash": stable_hash(fingerprint),
                    },
                )
            else:
                raise ValueError(f"Unknown pipeline stage: {stage}")

    # Submit attack jobs in priority order (across all pipelines).
    if pending_attacks:
        def _attack_sort_key(p):
            try:
                pri = attack_priority.index(p["attack_name"])
            except ValueError:
                pri = len(attack_priority)  # unlisted attacks last
            return (pri, p["pipeline_name"], p["defense_name"])
        pending_attacks.sort(key=_attack_sort_key)
        if attack_priority:
            ordered = [p["attack_name"] for p in pending_attacks]
            print(f"Submitting {len(pending_attacks)} attack jobs in priority order: {attack_priority} -> "
                  f"first 6={ordered[:6]} ... last 3={ordered[-3:]}")
        for p in pending_attacks:
            job_id = backend.submit(
                name=p["job_name"],
                command=p["wrapped"],
                time=get_time(cluster, "attack", "05:00:00"),
                output_log=str(p["stdout_log"]),
                error_log=str(p["stderr_log"]),
                depends_on=p["deps"],
            )
            append_submission_record(
                jobs_path,
                {
                    "job_id": job_id,
                    "job_name": p["job_name"],
                    "stage": "attack",
                    "pipeline": p["pipeline_name"],
                    "defense": p["defense_name"],
                    "attack": p["attack_name"],
                    "stage_dir": str(p["stage_dir"]),
                    "stdout_log": str(p["stdout_log"]),
                    "stderr_log": str(p["stderr_log"]),
                    "depends_on": p["deps"],
                    "fingerprint_hash": stable_hash(p["fingerprint"]),
                },
            )

    print("Experiment submission complete.")
    if args.backend == "local_gpu":
        backend.wait_all()
        print("All jobs finished.")


if __name__ == "__main__":
    main()
