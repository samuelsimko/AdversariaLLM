import subprocess
from pathlib import Path
import shlex
from typing import List, Optional
from .base import Backend


class SlurmBackend(Backend):
    def __init__(
        self,
        partition,
        account,
        gres=None,
        *,
        workdir: str,
        env_file: str | None = None,
        venv_activate: str | None = None,
        cpus_per_task: int | None = None,
        mem: str | None = None,
        mem_per_cpu: str | None = None,
        qos: str | None = None,
        constraint: str | None = None,
        gpus: str | int | None = None,
    ):
        self.partition = partition
        self.account = account
        self.gres = gres
        self.workdir = workdir
        self.env_file = env_file
        self.venv_activate = venv_activate
        self.cpus_per_task = cpus_per_task
        self.mem = mem
        self.mem_per_cpu = mem_per_cpu
        self.qos = qos
        self.constraint = constraint
        self.gpus = gpus

    def submit(
        self,
        *,
        name: str,
        command: List[str],
        time: str,
        output_log: str,
        error_log: str,
        depends_on: Optional[List[str]] = None,
        gres_override: Optional[str] = None,
    ) -> str:
        sbatch_cmd = [
            "sbatch",
            "--parsable",
            "--account", self.account,
            "--time", time,
            "--job-name", name,
            "--output", output_log,
            "--error", error_log,
        ]
        if self.partition:
            sbatch_cmd += ["-p", self.partition]
        # gpus can be either an int ("--gpus=1") or "<type>:N" ("--gpus=nvidia_a100_80gb_pcie:1").
        # On Euler the type-form acts as a partition router: slurm picks a partition that
        # *has* that GPU type (e.g. gpupr.4h), but inside that partition you may still get a
        # sibling SKU (e.g. A100-40GB instead of 80GB). That's fine for our compatibility
        # bound (sm_80 != sm_120 Blackwell). The `--gres=gpu:<type>:N` form is stricter but
        # requires a matching partition pin; without it Euler rejects with "Requested node
        # configuration is not available". Emit --gpus and let the user pin partition if
        # they need stricter typing.
        if self.gpus is not None:
            sbatch_cmd += [f"--gpus={self.gpus}"]
        # Per-call gres override (e.g. attack stages want gpumem:40g, train wants 80g).
        effective_gres = gres_override if gres_override is not None else self.gres
        if effective_gres:
            sbatch_cmd += ["--gres", effective_gres]
        if self.cpus_per_task is not None:
            sbatch_cmd += ["--cpus-per-task", str(self.cpus_per_task)]
        if self.mem is not None:
            sbatch_cmd += ["--mem", str(self.mem)]
        if self.mem_per_cpu is not None:
            sbatch_cmd += ["--mem-per-cpu", str(self.mem_per_cpu)]
        if self.qos is not None:
            sbatch_cmd += ["--qos", self.qos]
        if self.constraint is not None:
            sbatch_cmd += ["--constraint", self.constraint]

        if depends_on:
            sbatch_cmd.append(
                f"--dependency=afterok:{':'.join(depends_on)}"
            )

        shell_steps = [
            "set -euo pipefail",
            f"cd {shlex.quote(self.workdir)}",
        ]
        if self.venv_activate and Path(self.venv_activate).exists():
            shell_steps.append(f"source {shlex.quote(self.venv_activate)}")
        if self.env_file and Path(self.env_file).exists():
            shell_steps.append(f"source {shlex.quote(self.env_file)}")
        shell_steps.extend(
            [
                "nvidia-smi",
                "sleep 3",
                shlex.join(command),
            ]
        )
        wrapped = "bash -lc " + shlex.quote("; ".join(shell_steps))

        sbatch_cmd += ["--wrap", wrapped]

        print("▶ [SLURM]", " ".join(sbatch_cmd))
        job_id = subprocess.check_output(sbatch_cmd).decode().strip()
        print(f"  ↳ job_id={job_id}")
        return job_id
