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
        gres,
        *,
        workdir: str,
        env_file: str | None = None,
        venv_activate: str | None = None,
        cpus_per_task: int | None = None,
        mem: str | None = None,
        qos: str | None = None,
        constraint: str | None = None,
    ):
        self.partition = partition
        self.account = account
        self.gres = gres
        self.workdir = workdir
        self.env_file = env_file
        self.venv_activate = venv_activate
        self.cpus_per_task = cpus_per_task
        self.mem = mem
        self.qos = qos
        self.constraint = constraint

    def submit(
        self,
        *,
        name: str,
        command: List[str],
        time: str,
        output_log: str,
        error_log: str,
        depends_on: Optional[List[str]] = None,
    ) -> str:
        sbatch_cmd = [
            "sbatch",
            "--parsable",
            "-p", self.partition,
            "--account", self.account,
            "--gres", self.gres,
            "--time", time,
            "--job-name", name,
            "--output", output_log,
            "--error", error_log,
        ]
        if self.cpus_per_task is not None:
            sbatch_cmd += ["--cpus-per-task", str(self.cpus_per_task)]
        if self.mem is not None:
            sbatch_cmd += ["--mem", str(self.mem)]
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
