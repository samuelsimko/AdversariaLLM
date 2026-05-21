"""Template jailbreak attack.

For each harmful behavior, sample N jailbreak templates (PyRIT vendored set under
`data/pyrit_templates/` by default) and substitute the behavior into the template's
`{{ prompt }}` slot. Each rendered template is one attack step.

This is a discrete, non-optimization attack — it just enumerates known templates
and lets the judge sort out which ones got through.

Reference: Microsoft PyRIT — https://github.com/microsoft/PyRIT (MIT-licensed)
"""

from __future__ import annotations

import copy
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers
import yaml

from .attack import (
    Attack,
    AttackResult,
    AttackStepResult,
    GenerationConfig,
    SingleAttackRunResult,
)
from ..lm_utils import generate_ragged_batched, get_losses_batched, prepare_conversation
from ..types import Conversation


# Match {{ prompt }} / {{prompt}} / {prompt} — all variants seen in the PyRIT set.
_PLACEHOLDER_RE = re.compile(r"\{\{\s*prompt\s*\}\}|\{prompt\}")


@dataclass
class TemplateJailbreakConfig:
    """Config for the template-jailbreak attack."""

    name: str = "template_jailbreak"
    type: str = "discrete"
    version: str = "0.0.1"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0

    # Where to load templates from. Templates are YAML files (PyRIT format):
    #   parameters: [prompt]
    #   value: >
    #     ... {{ prompt }} ...
    templates_dir: str = "data/pyrit_templates"

    # Glob (relative to templates_dir) for which YAMLs to include.
    # Default = only the 90 top-level single-parameter templates; subdirs
    # (Arth_Singh, multi_parameter, pliny) are skipped by default because some
    # are multi-parameter and require extra slot wiring.
    template_glob: str = "*.yaml"

    # Per-behavior subsample size. None = use *all* matching templates.
    num_templates: int | None = 20

    # If True, drop templates whose `value` doesn't contain a {{ prompt }} slot.
    require_prompt_placeholder: bool = True

    # If True, every behavior sees the same subsample (seeded by `seed` only).
    # If False, the subsample is drawn per-behavior (seeded by `seed + idx`),
    # which gives different templates per behavior.
    fixed_subsample: bool = False


def _load_templates(cfg: TemplateJailbreakConfig) -> list[dict]:
    """Load templates from disk. Returns list of {name, value} dicts."""
    base = Path(cfg.templates_dir)
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    files = sorted(base.glob(cfg.template_glob))
    out: list[dict] = []
    for f in files:
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"[template_jailbreak] skipping unparseable template {f.name}: {e}")
            continue
        if not isinstance(data, dict):
            continue
        value = data.get("value")
        if not isinstance(value, str):
            continue
        if cfg.require_prompt_placeholder and not _PLACEHOLDER_RE.search(value):
            continue
        out.append({"name": data.get("name") or f.stem, "value": value, "path": str(f)})
    if not out:
        raise ValueError(
            f"No usable templates found under {base} matching {cfg.template_glob!r}. "
            "Did you forget to vendor data/pyrit_templates/?"
        )
    logging.info(f"[template_jailbreak] loaded {len(out)} templates from {base}")
    return out


def _render(template_value: str, harmful_prompt: str) -> str:
    """Substitute {{ prompt }} (and variants) in the template with harmful_prompt.

    Use a callable replacement so backslash sequences in `harmful_prompt`
    (e.g. \\U, \\1) are treated as literal characters rather than regex
    backreference/escape syntax. Otherwise re.sub raises `re.error: bad
    escape` on certain HarmBench behaviors.
    """
    return _PLACEHOLDER_RE.sub(lambda _m: harmful_prompt, template_value)


class TemplateJailbreakAttack(Attack):
    """Render each behavior into N jailbreak templates and treat each as a step."""

    def __init__(self, config: TemplateJailbreakConfig):
        super().__init__(config)
        self._templates: list[dict] | None = None  # lazy

    @torch.no_grad
    def run(
        self,
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer,
        dataset: torch.utils.data.Dataset,
    ) -> AttackResult:
        t0 = time.time()
        if self._templates is None:
            self._templates = _load_templates(self.config)
        templates = self._templates

        cfg = self.config
        gen_cfg = cfg.generation_config
        num_per_behavior = (
            len(templates) if cfg.num_templates is None
            else min(cfg.num_templates, len(templates))
        )

        # Pre-pick template indices per behavior to keep the run reproducible.
        if cfg.fixed_subsample:
            rng = random.Random(cfg.seed)
            fixed_idx = rng.sample(range(len(templates)), num_per_behavior)
            per_behavior_idx = [list(fixed_idx) for _ in range(len(dataset))]
        else:
            per_behavior_idx = []
            for i in range(len(dataset)):
                rng = random.Random(cfg.seed + i)
                per_behavior_idx.append(rng.sample(range(len(templates)), num_per_behavior))

        runs: list[SingleAttackRunResult] = []
        for behavior_i, conversation in enumerate(dataset):
            assert len(conversation) == 2, "template_jailbreak assumes single-turn conversation."
            original_prompt: Conversation = conversation
            harmful_prompt = conversation[0]["content"]
            target = conversation[-1]["content"]

            picks = per_behavior_idx[behavior_i]
            # Build all rendered conversations + their token tensors up front, then
            # batch the loss/generation calls — same shape as direct.py uses.
            rendered_convs: list[Conversation] = []
            full_token_tensors: list[torch.Tensor] = []
            prompt_token_tensors: list[torch.Tensor] = []
            for tidx in picks:
                tpl = templates[tidx]
                rendered_user = _render(tpl["value"], harmful_prompt)
                conv: Conversation = [
                    {"role": "user", "content": rendered_user},
                    {"role": "assistant", "content": target},
                ]
                rendered_convs.append(conv)
                token_tensors = prepare_conversation(tokenizer, conv)
                flat_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]
                full_token_tensors.append(torch.cat(flat_tokens, dim=0))
                prompt_token_tensors.append(torch.cat(flat_tokens[:-1]))

            # Loss per template (avg over target tokens only)
            shifted = [t.roll(-1, 0) for t in full_token_tensors]
            all_losses = get_losses_batched(
                model,
                targets=shifted,
                token_list=full_token_tensors,
                initial_batch_size=len(picks),
            )
            instance_losses: list[float | None] = []
            for j in range(len(picks)):
                full_len = full_token_tensors[j].size(0)
                prompt_len = prompt_token_tensors[j].size(0)
                tgt_losses = all_losses[j][prompt_len - 1 : full_len - 1]
                instance_losses.append(tgt_losses.mean().item() if tgt_losses.numel() > 0 else None)

            # Generate completions for every rendered template (batched).
            completions = generate_ragged_batched(
                model,
                tokenizer,
                token_list=prompt_token_tensors,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                top_k=gen_cfg.top_k,
                num_return_sequences=gen_cfg.num_return_sequences,
                initial_batch_size=len(picks) * gen_cfg.num_return_sequences,
            )

            steps: list[AttackStepResult] = []
            for j, tidx in enumerate(picks):
                model_input = copy.deepcopy(rendered_convs[j])
                # Mirror direct.py's convention: blank the assistant turn so
                # downstream judging treats the user turn as the prompt.
                model_input[-1]["content"] = ""
                steps.append(
                    AttackStepResult(
                        step=j,
                        model_completions=completions[j],
                        time_taken=0.0,
                        loss=instance_losses[j],
                        flops=0,
                        model_input=model_input,
                        model_input_tokens=prompt_token_tensors[j].tolist(),
                    )
                )

            t_now = time.time()
            runs.append(SingleAttackRunResult(
                original_prompt=original_prompt,
                steps=steps,
                total_time=t_now - t0,
            ))
            logging.info(
                f"[template_jailbreak] behavior {behavior_i + 1}/{len(dataset)} "
                f"done; {len(steps)} templates rendered"
            )

        logging.info(f"[template_jailbreak] total time {time.time() - t0:.1f}s")
        return AttackResult(runs=runs)
