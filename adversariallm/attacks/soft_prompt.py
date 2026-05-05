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

    # Batched optimization. When >1, the attack groups prompts whose per-iter
    # config matches on (lr, num_tokens, num_steps, optim_str_init, rand_init,
    # early_stop_loss) and optimizes each sub-batch jointly with one Adam.
    # Per-row done-mask handles early-stop. Generation is batched too.
    # Set to 1 to keep the original sequential behavior.
    batch_size: int = 1

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


@dataclass
class SoftPromptBatchedOptResult:
    """Per-row results from a batched soft-prompt optimization."""
    losses_per_row: list[list[float]]      # per-row per-step loss
    optim_embeds_per_row: list[torch.Tensor]   # [num_tokens, hidden] per row, on CPU
    input_embeds_per_row: list[torch.Tensor]   # [before+optim+after, hidden] per row, on CPU


def run_soft_opt_batched(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    items: list[tuple[Union[str, list[dict[str, str]]], str]],
    config: SoftPromptConfig,
    device: torch.device,
) -> SoftPromptBatchedOptResult:
    """Optimize len(items) soft prompts jointly under a single Adam.

    All rows MUST share the same (num_tokens, num_steps, lr, rand_init,
    optim_str_init, early_stop_loss) — i.e., one bucket. Per-row early-stop is
    handled via a done-mask: rows that have hit early_stop_loss freeze their
    optim slot to the snapshot taken at the threshold-crossing step, while
    other rows keep optimizing. Optimization halts when all rows are done or
    num_steps is reached.
    """
    if not items:
        return SoftPromptBatchedOptResult([], [], [])

    model.enable_input_require_grads()
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    emb = model.get_input_embeddings()
    dtype = emb.weight.dtype
    H = model.config.hidden_size
    B = len(items)

    # Tokenize before / after / target for every row.
    before_ids_list: list[torch.Tensor] = []
    after_ids_list: list[torch.Tensor] = []
    target_ids_list: list[torch.Tensor] = []
    for messages, target in items:
        msgs = _ensure_messages(messages)
        if not msgs:
            raise ValueError("each batched item needs at least one prompt message")
        if not any("{optim_str}" in d["content"] for d in msgs):
            msgs[-1]["content"] = msgs[-1]["content"] + "{optim_str}"
        template = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "", 1)
        if "{optim_str}" not in template:
            raise ValueError("Chat template did not preserve {optim_str} placeholder.")
        before_str, after_str = template.split("{optim_str}", 1)
        tgt = (" " + target) if config.add_space_before_target else target
        before_ids_list.append(
            tokenizer(before_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
        )
        after_ids_list.append(
            tokenizer(after_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
        )
        tgt_ids = tokenizer(tgt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
        if tgt_ids.numel() == 0:
            raise ValueError("Target tokenization is empty.")
        target_ids_list.append(tgt_ids)

    target_lens = torch.tensor([t.numel() for t in target_ids_list], device=device)

    # Initialize optim_embeds: shared shape [B, num_tokens, H].
    if not config.rand_init:
        optim_ids = tokenizer(
            config.optim_str_init, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)
        base_optim = emb(optim_ids).detach().to(dtype)  # [1, num_tokens, H]
        if base_optim.shape[1] != config.num_tokens:
            # Fall back to declared num_tokens if the init string tokenizes differently.
            num_tokens = base_optim.shape[1]
        else:
            num_tokens = config.num_tokens
        optim_embeds = base_optim.expand(B, -1, -1).clone().requires_grad_(True)
    else:
        num_tokens = config.num_tokens
        optim_embeds = torch.randn(B, num_tokens, H, device=device, dtype=dtype, requires_grad=True)

    # Cache fixed embedding pieces per row.
    before_embs = [emb(t).detach().to(dtype) for t in before_ids_list]
    after_embs = [emb(t).detach().to(dtype) for t in after_ids_list]
    target_embs = [emb(t).detach().to(dtype) for t in target_ids_list]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_emb = emb(torch.tensor([pad_id], device=device)).detach().to(dtype).squeeze(0)  # [H]

    seq_lens = torch.tensor(
        [b.shape[0] + num_tokens + a.shape[0] + tg.shape[0]
         for b, a, tg in zip(before_embs, after_embs, target_embs)],
        device=device,
    )
    T = int(seq_lens.max().item())

    optim = torch.optim.Adam([optim_embeds], lr=config.lr)

    losses_per_row: list[list[float]] = [[] for _ in range(B)]
    done_mask = torch.zeros(B, dtype=torch.bool, device=device)
    # Snapshot of optim_embeds[i] taken at the step row i first satisfied early_stop.
    final_optim = [None] * B
    final_loss = [None] * B

    for step in range(config.num_steps):
        optim.zero_grad(set_to_none=True)

        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        target_pos: list[torch.Tensor] = []
        for i in range(B):
            content = torch.cat(
                [before_embs[i], optim_embeds[i], after_embs[i], target_embs[i]], dim=0
            )
            L = content.shape[0]
            pad_n = T - L
            if pad_n > 0:
                row = torch.cat([pad_emb.unsqueeze(0).expand(pad_n, H), content], dim=0)
            else:
                row = content
            rows.append(row.unsqueeze(0))
            mask = torch.zeros(T, dtype=torch.long, device=device)
            mask[pad_n:] = 1
            masks.append(mask.unsqueeze(0))
            tlen = int(target_lens[i].item())
            target_pos.append(torch.arange(T - tlen, T, device=device))

        inputs_embeds = torch.cat(rows, dim=0)
        attention_mask = torch.cat(masks, dim=0)

        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # [B, T, V]

        per_row_losses = []
        for i in range(B):
            pos = target_pos[i]
            shift_logits = logits[i, pos - 1, :]
            ce = F.cross_entropy(shift_logits, target_ids_list[i])
            per_row_losses.append(ce)
        per_row_t = torch.stack(per_row_losses)  # [B]
        active = (~done_mask).float()
        active_count = active.sum().item()
        if active_count <= 0:
            break
        # Sum (NOT mean) of active rows. With mean, d(loss)/d(optim_embeds[i])
        # would be 1/active_count of the sequential gradient — effectively
        # shrinking the LR by the bucket size and slowing convergence. Sum
        # keeps each row's gradient magnitude identical to the sequential path.
        loss = (per_row_t * active).sum()

        for i in range(B):
            li = per_row_losses[i].detach().item()
            losses_per_row[i].append(li)
            if not bool(done_mask[i].item()):
                final_loss[i] = li
                if config.early_stop_loss and li < config.early_stop_loss:
                    final_optim[i] = optim_embeds[i].detach().clone()
                    done_mask[i] = True

        if config.verbose:
            logging.info(
                "[soft_prompt:batched] step=%d active=%d mean_loss=%.6f",
                step, int(active_count), float(loss.item()),
            )

        if bool(done_mask.all().item()):
            break

        loss.backward()
        optim.step()

    # For rows that never hit early_stop, use the last optim_embeds.
    for i in range(B):
        if final_optim[i] is None:
            final_optim[i] = optim_embeds[i].detach().clone()

    # Build per-row generation inputs (before + optim + after, no target, no padding).
    input_embeds_per_row: list[torch.Tensor] = []
    optim_embeds_per_row: list[torch.Tensor] = []
    for i in range(B):
        gen = torch.cat([before_embs[i], final_optim[i], after_embs[i]], dim=0)
        input_embeds_per_row.append(gen.detach().cpu())
        optim_embeds_per_row.append(final_optim[i].detach().cpu())

    return SoftPromptBatchedOptResult(
        losses_per_row=losses_per_row,
        optim_embeds_per_row=optim_embeds_per_row,
        input_embeds_per_row=input_embeds_per_row,
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

    @torch.no_grad()
    def _generate_from_embeds_batched(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        input_embeds_per_row: list[torch.Tensor],
    ) -> list[list[str]]:
        """Batch generate from a list of variable-length per-row embeddings.

        Returns a list of length len(input_embeds_per_row); each entry is the
        list of decoded completions for that row (length =
        generation_config.num_return_sequences).
        """
        if not input_embeds_per_row:
            return []
        device = model.device
        dtype = model.dtype
        H = input_embeds_per_row[0].shape[-1]
        B = len(input_embeds_per_row)
        lens = [int(x.shape[0]) for x in input_embeds_per_row]
        T = max(lens)

        emb = model.get_input_embeddings()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_emb = emb(torch.tensor([pad_id], device=device)).detach().to(dtype).squeeze(0)

        rows = []
        masks = []
        for x, L in zip(input_embeds_per_row, lens):
            x = x.to(device=device, dtype=dtype)
            pad_n = T - L
            if pad_n > 0:
                row = torch.cat([pad_emb.unsqueeze(0).expand(pad_n, H), x], dim=0)
            else:
                row = x
            rows.append(row.unsqueeze(0))
            mask = torch.zeros(T, dtype=torch.long, device=device)
            mask[pad_n:] = 1
            masks.append(mask.unsqueeze(0))
        inputs_embeds = torch.cat(rows, dim=0)
        attention_mask = torch.cat(masks, dim=0)

        gen_ids = model.generate(
            inputs_embeds=inputs_embeds,
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
        # gen_ids has shape [B * num_return_sequences, max_new_tokens] when using
        # inputs_embeds (HF returns only the new tokens).
        nrs = self.config.generation_config.num_return_sequences
        out: list[list[str]] = []
        for i in range(B):
            row_ids = gen_ids[i * nrs : (i + 1) * nrs]
            out.append([tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in row_ids])
        return out

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

    def _resolve_iter_config(self, prompt_index: int) -> SoftPromptConfig:
        """Build the per-iteration config (applying per-prompt randomization)."""
        if self.config.random_config_per_prompt:
            pick = pick_pool_config(self.config, prompt_index)
            ic = copy.copy(self.config)
            ic.lr = float(pick["lr"])
            ic.optim_str_init = str(pick["optim_str_init"])
            ic.early_stop_loss = float(pick["early_stop_loss"])
            ic.rand_init = False
            logging.info(
                "[soft_prompt] prompt_index=%d POOL lr=%.5g early_stop=%.5g init=%r",
                prompt_index, ic.lr, ic.early_stop_loss, ic.optim_str_init,
            )
            return ic
        if self.config.randomize_per_prompt:
            draw = derive_randomized_soft_prompt_params(self.config, prompt_index)
            ic = copy.copy(self.config)
            ic.num_tokens = draw["num_tokens"]
            ic.num_steps = draw["num_steps"]
            ic.lr = float(draw["lr"])
            if self.config.randomize_force_rand_init:
                ic.rand_init = True
            logging.info(
                "[soft_prompt] prompt_index=%d num_tokens=%d num_steps=%d lr=%.5g",
                prompt_index, ic.num_tokens, ic.num_steps, ic.lr,
            )
            return ic
        return self.config

    def _bucket_key(self, ic: SoftPromptConfig) -> tuple:
        """Configs that share this key can be batched in one Adam optimization."""
        return (
            float(ic.lr),
            int(ic.num_tokens),
            int(ic.num_steps),
            bool(ic.rand_init),
            ic.optim_str_init,
            float(ic.early_stop_loss),
        )

    def run(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: PromptDataset,
    ) -> AttackResult:
        if not self._initialized:
            self.initialize(model, tokenizer)

        device = model.device
        prompt_indices = self._resolve_prompt_indices(dataset)

        # Pre-resolve per-iteration configs and the per-prompt content.
        conversations = [conv for conv in dataset]
        for conv in conversations:
            if not conv or conv[-1]["role"] != "assistant":
                raise ValueError("SoftPromptAttack expects each example to end with an assistant target turn.")

        if int(getattr(self.config, "batch_size", 1) or 1) > 1:
            return self._run_batched(model, tokenizer, conversations, prompt_indices, device)
        return self._run_sequential(model, tokenizer, conversations, prompt_indices, device)

    def _run_sequential(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        conversations: list,
        prompt_indices: list[int],
        device: torch.device,
    ) -> AttackResult:
        runs: list[SingleAttackRunResult] = []
        for iter_index, conversation in enumerate(conversations):
            t0 = time.time()
            prompt_messages = copy.deepcopy(conversation[:-1])
            target = conversation[-1]["content"]
            iter_config = self._resolve_iter_config(prompt_indices[iter_index])

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

    def _run_batched(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        conversations: list,
        prompt_indices: list[int],
        device: torch.device,
    ) -> AttackResult:
        N = len(conversations)
        iter_configs = [self._resolve_iter_config(prompt_indices[i]) for i in range(N)]
        # Bucket by config so we can batch compatible rows together.
        buckets: dict[tuple, list[int]] = {}
        for i, ic in enumerate(iter_configs):
            buckets.setdefault(self._bucket_key(ic), []).append(i)

        bs = int(self.config.batch_size)
        # Pre-allocate the per-prompt result slots so we can fill in any order.
        steps_by_index: list[AttackStepResult | None] = [None] * N

        for key, indices in buckets.items():
            for sub_start in range(0, len(indices), bs):
                sub = indices[sub_start : sub_start + bs]
                t0 = time.time()
                bucket_cfg = iter_configs[sub[0]]  # all rows in `sub` share the same config
                items: list[tuple[Any, str]] = []
                for i in sub:
                    items.append((copy.deepcopy(conversations[i][:-1]), conversations[i][-1]["content"]))
                logging.info(
                    "[soft_prompt:batched] sub_batch_size=%d lr=%.5g num_tokens=%d num_steps=%d "
                    "early_stop=%.5g rand_init=%s init=%r prompt_indices=%s",
                    len(sub), bucket_cfg.lr, bucket_cfg.num_tokens, bucket_cfg.num_steps,
                    bucket_cfg.early_stop_loss, bucket_cfg.rand_init, bucket_cfg.optim_str_init,
                    [prompt_indices[i] for i in sub],
                )

                try:
                    res = run_soft_opt_batched(
                        model=model, tokenizer=tokenizer, items=items,
                        config=bucket_cfg, device=device,
                    )
                    completions_per_row = self._generate_from_embeds_batched(
                        model, tokenizer, res.input_embeds_per_row,
                    )
                except Exception:
                    logging.exception(
                        "SoftPromptAttack batched failed for prompt indices: %s",
                        [prompt_indices[i] for i in sub],
                    )
                    res = None
                    completions_per_row = [[""] for _ in sub]

                wall = time.time() - t0
                # Distribute the wall time evenly across the rows in the sub-batch.
                per_row_t = wall / max(len(sub), 1)
                for k, i in enumerate(sub):
                    prompt_messages = copy.deepcopy(conversations[i][:-1])
                    model_input = copy.deepcopy(prompt_messages)
                    model_input.append({"role": "assistant", "content": ""})
                    if res is not None:
                        last_loss = res.losses_per_row[k][-1] if res.losses_per_row[k] else None
                        embeds = res.input_embeds_per_row[k]
                    else:
                        last_loss = None
                        embeds = None
                    steps_by_index[i] = AttackStepResult(
                        step=0,
                        model_completions=completions_per_row[k],
                        time_taken=per_row_t,
                        loss=last_loss,
                        flops=None,
                        model_input=model_input,
                        model_input_tokens=None,
                        model_input_embeddings=embeds,
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        runs: list[SingleAttackRunResult] = []
        for i, conv in enumerate(conversations):
            step = steps_by_index[i]
            runs.append(
                SingleAttackRunResult(
                    original_prompt=copy.deepcopy(conv),
                    steps=[step] if step is not None else [],
                    total_time=step.time_taken if step is not None else 0.0,
                )
            )
        return AttackResult(runs=runs)
