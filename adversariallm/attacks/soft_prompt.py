import copy
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
import tqdm
import transformers

from ..dataset import PromptDataset
from ..types import Conversation
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult


@dataclass
class SoftPromptConfig:
    name: str = "soft_prompt"
    type: str = "embedding"
    version: str = "0.0.1"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: Optional[int] = 0
    num_steps: int = 200
    optim_str_init: str = "x " * 5
    rand_init: bool = False
    num_tokens: int = 5
    lr: float = 0.01
    early_stop_loss: float = 0.001
    add_space_before_target: bool = False
    verbose: bool = False
    extra_fields: dict[str, Any] = field(default_factory=dict)

    # Per-prompt hyperparameter randomization. When randomize_per_prompt=True,
    # for each prompt index `i` in the attacked dataset we deterministically
    # draw (num_tokens, num_steps, lr) using a Random seeded by
    # randomize_seed + i, overriding the static fields above. The same prompt
    # index across DIFFERENT models gets the SAME hyperparams (so we can
    # compare cell-vs-cell at matched attack difficulty), while different
    # indices get different ones (so we sweep the attack hyperparam space).
    randomize_per_prompt: bool = False
    randomize_seed: int = 2024
    randomize_force_rand_init: bool = True   # ignore optim_str_init when randomizing
    num_tokens_min: int = 1
    num_tokens_max: int = 10
    num_steps_min: int = 100
    num_steps_max: int = 700
    # Log-uniform draw for lr. Defaults span 1e-4 .. 1e-1 (3 orders of magnitude).
    lr_log10_min: float = -4.0
    lr_log10_max: float = -1.0

    # Discrete config pool. When random_config_per_prompt=True, each prompt
    # index deterministically draws one entry from `config_pool`, overriding
    # the static (lr, optim_str_init, early_stop_loss). Each entry is a dict
    # with keys {"lr", "optim_str_init", "early_stop_loss"}. The init string
    # is tokenized to determine num_tokens. num_steps stays at the static
    # field (typically 1000 — early_stop terminates earlier when loss
    # crosses the threshold). Seeded by randomize_seed + prompt_index.
    random_config_per_prompt: bool = False
    config_pool: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SoftPromptConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        init_data = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        if "str_length" in extra and "optim_str_init" not in init_data:
            init_data["optim_str_init"] = "x " * int(extra["str_length"])
        init_data["extra_fields"] = data
        return cls(**init_data)


def derive_randomized_soft_prompt_params(
    base_config: SoftPromptConfig,
    prompt_index: int,
) -> dict[str, Any]:
    """Deterministically draw per-prompt soft-prompt hyperparameters.

    Same prompt_index always produces the same draw (across models / runs)
    because the RNG is seeded by ``randomize_seed + prompt_index`` only.
    """
    rng = random.Random(int(base_config.randomize_seed) + int(prompt_index))
    num_tokens = rng.randint(base_config.num_tokens_min, base_config.num_tokens_max)
    num_steps = rng.randint(base_config.num_steps_min, base_config.num_steps_max)
    lr = 10.0 ** rng.uniform(base_config.lr_log10_min, base_config.lr_log10_max)
    return {"num_tokens": num_tokens, "num_steps": num_steps, "lr": lr}


def pick_pool_config(
    base_config: SoftPromptConfig,
    prompt_index: int,
) -> dict[str, Any]:
    """Deterministically pick one entry from ``config_pool`` for this prompt.

    Same RNG seed convention as ``derive_randomized_soft_prompt_params``: the
    same prompt_index always picks the same pool entry across models/runs.
    """
    if not base_config.config_pool:
        raise ValueError("random_config_per_prompt=True but config_pool is empty.")
    rng = random.Random(int(base_config.randomize_seed) + int(prompt_index))
    entry = dict(rng.choice(list(base_config.config_pool)))
    # Sanity check required keys
    for k in ("lr", "optim_str_init", "early_stop_loss"):
        if k not in entry:
            raise ValueError(f"config_pool entry missing key {k!r}: {entry}")
    return entry


@dataclass
class SoftPromptOptResult:
    losses: list[float]
    optim_embeds: torch.Tensor
    input_embeds: torch.Tensor


def _ensure_messages(messages: Union[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return copy.deepcopy(messages)


def run_soft_opt(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: Union[str, list[dict[str, str]]],
    target: str,
    config: SoftPromptConfig,
    device: torch.device,
) -> SoftPromptOptResult:
    model.enable_input_require_grads()

    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    msgs = _ensure_messages(messages)
    if not msgs:
        raise ValueError("Soft prompt attack requires at least one prompt message.")
    if not any("{optim_str}" in d["content"] for d in msgs):
        msgs[-1]["content"] = msgs[-1]["content"] + "{optim_str}"

    template = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(tokenizer.bos_token, "", 1)

    if "{optim_str}" not in template:
        raise ValueError("Chat template did not preserve {optim_str} placeholder.")

    before_str, after_str = template.split("{optim_str}", 1)
    tgt = (" " + target) if config.add_space_before_target else target

    before_ids = tokenizer(before_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    after_ids = tokenizer(after_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    target_ids = tokenizer(tgt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if target_ids.numel() == 0:
        raise ValueError("Target tokenization is empty.")

    emb = model.get_input_embeddings()
    # Detach the fixed embeddings so backward through them doesn't try to
    # walk back into the embedding-table's already-freed graph in iter 2+.
    before_embeds = emb(before_ids).detach()
    after_embeds = emb(after_ids).detach()
    target_embeds = emb(target_ids).detach()

    if not config.rand_init:
        optim_ids = tokenizer(config.optim_str_init, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        optim_embeds = emb(optim_ids).detach().clone().requires_grad_(True)
    else:
        optim_embeds = torch.randn((1, config.num_tokens, model.config.hidden_size), device=device).requires_grad_(True)

    dtype = emb.weight.dtype
    optim_embeds = optim_embeds.to(dtype).detach().requires_grad_(True)
    before_embeds = before_embeds.to(dtype)
    after_embeds = after_embeds.to(dtype)
    target_embeds = target_embeds.to(dtype)

    opt = torch.optim.Adam([optim_embeds], lr=config.lr)
    losses: list[float] = []

    for step in tqdm.tqdm(range(config.num_steps), disable=not config.verbose):
        opt.zero_grad(set_to_none=True)

        train_embeds = torch.cat([before_embeds, optim_embeds, after_embeds, target_embeds.detach()], dim=1)
        attention_mask = torch.ones(train_embeds.shape[:2], dtype=torch.long, device=device)
        out = model(inputs_embeds=train_embeds, attention_mask=attention_mask, use_cache=False)
        logits = out.logits

        shift = train_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : shift - 1 + target_ids.shape[1], :].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            target_ids.view(-1),
        )

        losses.append(float(loss.item()))
        if config.verbose:
            logging.info("[soft_prompt] step=%d loss=%.6f", step, loss.item())

        if config.early_stop_loss and loss.item() < config.early_stop_loss:
            break

        loss.backward()
        opt.step()

    gen_input_embeds = torch.cat([before_embeds, optim_embeds.detach(), after_embeds], dim=1)

    return SoftPromptOptResult(
        losses=losses,
        optim_embeds=optim_embeds.detach().cpu(),
        input_embeds=gen_input_embeds.detach().cpu(),
    )


class SoftPromptAttack(Attack):
    def __init__(self, config: SoftPromptConfig):
        normalized = config if isinstance(config, SoftPromptConfig) else SoftPromptConfig.from_mapping(dict(config))
        super().__init__(normalized)
        self._initialized = False

    def initialize(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> None:
        model.eval()
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self._initialized = True

    @torch.no_grad()
    def _generate_from_embeds(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        input_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, list[str]]:
        input_embeds = input_embeds.to(device=model.device, dtype=model.dtype)
        attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=model.device)

        gen_ids = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            do_sample=self.config.generation_config.temperature > 0,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        texts = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in gen_ids]
        return gen_ids, texts

    def _resolve_prompt_indices(self, dataset: PromptDataset) -> list[int]:
        """Return the per-iteration prompt index used for randomization seeding.

        Prefers the dataset's own ``config_idx`` (the dataset-level behavior IDs
        from the YAML, e.g. ``[0, 1, ..., 49]``) so that the same behavior gets
        the same hyperparams across runs that re-slice the dataset. Falls back
        to a 0..N-1 enumeration when the dataset doesn't expose config_idx.
        """
        n = len(dataset)
        cfg_idx = getattr(dataset, "config_idx", None)
        if isinstance(cfg_idx, (list, tuple)) and len(cfg_idx) == n:
            return [int(x) for x in cfg_idx]
        if isinstance(cfg_idx, int) and n == 1:
            return [int(cfg_idx)]
        return list(range(n))

    def run(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: PromptDataset,
    ) -> AttackResult:
        if not self._initialized:
            self.initialize(model, tokenizer)

        runs: list[SingleAttackRunResult] = []
        device = model.device
        prompt_indices = self._resolve_prompt_indices(dataset)

        for iter_index, conversation in enumerate(dataset):
            t0 = time.time()
            if not conversation or conversation[-1]["role"] != "assistant":
                raise ValueError("SoftPromptAttack expects each example to end with an assistant target turn.")

            prompt_messages = copy.deepcopy(conversation[:-1])
            target = conversation[-1]["content"]

            # Per-prompt hyperparameter randomization: keep self.config immutable,
            # build a per-iteration override.
            iter_config = self.config
            randomized_meta: dict[str, Any] = {}
            if self.config.random_config_per_prompt:
                prompt_index = prompt_indices[iter_index]
                pick = pick_pool_config(self.config, prompt_index)
                iter_config = copy.copy(self.config)
                iter_config.lr = float(pick["lr"])
                iter_config.optim_str_init = str(pick["optim_str_init"])
                iter_config.early_stop_loss = float(pick["early_stop_loss"])
                iter_config.rand_init = False  # use the meaningful init string
                randomized_meta = {
                    "_pool_prompt_index": prompt_index,
                    "_pool_lr": iter_config.lr,
                    "_pool_optim_str_init": iter_config.optim_str_init,
                    "_pool_early_stop_loss": iter_config.early_stop_loss,
                }
                logging.info(
                    "[soft_prompt] prompt_index=%d POOL lr=%.5g early_stop=%.5g init=%r",
                    prompt_index,
                    iter_config.lr,
                    iter_config.early_stop_loss,
                    iter_config.optim_str_init,
                )
            elif self.config.randomize_per_prompt:
                prompt_index = prompt_indices[iter_index]
                draw = derive_randomized_soft_prompt_params(self.config, prompt_index)
                iter_config = copy.copy(self.config)
                iter_config.num_tokens = draw["num_tokens"]
                iter_config.num_steps = draw["num_steps"]
                iter_config.lr = float(draw["lr"])
                if self.config.randomize_force_rand_init:
                    iter_config.rand_init = True
                randomized_meta = {
                    "_randomized_prompt_index": prompt_index,
                    "_randomized_num_tokens": iter_config.num_tokens,
                    "_randomized_num_steps": iter_config.num_steps,
                    "_randomized_lr": iter_config.lr,
                }
                logging.info(
                    "[soft_prompt] prompt_index=%d num_tokens=%d num_steps=%d lr=%.5g",
                    prompt_index,
                    iter_config.num_tokens,
                    iter_config.num_steps,
                    iter_config.lr,
                )

            try:
                res = run_soft_opt(
                    model=model,
                    tokenizer=tokenizer,
                    messages=prompt_messages,
                    target=target,
                    config=iter_config,
                    device=device,
                )
                _, generated = self._generate_from_embeds(model, tokenizer, res.input_embeds)

                model_input = copy.deepcopy(prompt_messages)
                model_input.append({"role": "assistant", "content": ""})
                step = AttackStepResult(
                    step=0,
                    model_completions=generated,
                    time_taken=time.time() - t0,
                    loss=res.losses[-1] if res.losses else None,
                    flops=None,
                    model_input=model_input,
                    model_input_tokens=None,
                    model_input_embeddings=res.input_embeds,
                )
            except Exception:
                logging.exception("SoftPromptAttack failed on prompt: %s", conversation[0]["content"] if conversation else "")
                model_input = copy.deepcopy(prompt_messages)
                model_input.append({"role": "assistant", "content": ""})
                step = AttackStepResult(
                    step=0,
                    model_completions=[""],
                    time_taken=time.time() - t0,
                    loss=None,
                    flops=None,
                    model_input=model_input,
                    model_input_tokens=None,
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            runs.append(
                SingleAttackRunResult(
                    original_prompt=copy.deepcopy(conversation),
                    steps=[step],
                    total_time=time.time() - t0,
                )
            )

        return AttackResult(runs=runs)
