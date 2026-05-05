import argparse
import csv
import inspect
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_MAX_STEPS = 1500
DEFAULT_LR = 2e-4
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_ULTRACHAT_SAMPLES = 5000
DEFAULT_CB_LIMIT = 5000
DEFAULT_PAIR_LIMIT = 5000
DEFAULT_HARM_CE_MIN = 5.0
DEFAULT_W_BENIGN = 1.0
DEFAULT_W_HARM = 1.0
DEFAULT_W_KL = 0.1
DEFAULT_W_JEPA = 0.5
DEFAULT_ALIGN_LAYER = 25
DEFAULT_REPORT_TO = "wandb"
DEFAULT_SEED = 42
DEFAULT_REP_LAYERS = list(range(20, 36))

DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05


def _chat_template_input_ids(chat_tokens):
    if isinstance(chat_tokens, torch.Tensor):
        return chat_tokens
    if hasattr(chat_tokens, "input_ids"):
        return chat_tokens.input_ids
    if isinstance(chat_tokens, dict) and "input_ids" in chat_tokens:
        return chat_tokens["input_ids"]
    raise TypeError(f"Unsupported chat template output type: {type(chat_tokens)!r}")


def _first_present(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "")
                parts.append(f"{role}: {content}" if role else str(content))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(value).strip()


def tokenize_chat_generic(prompts, responses, tokenizer, max_length=256):
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have matching lengths")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = []
    for prompt, response in zip(prompts, responses):
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        full = _chat_template_input_ids(full)

        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_generation_prompt=True,
        )
        prompt_only = _chat_template_input_ids(prompt_only)

        labels = full.clone()
        prompt_len = min(prompt_only.shape[-1], full.shape[-1])
        labels[0, :prompt_len] = -100

        full = full[:, :max_length]
        labels = labels[:, :max_length]
        pad = max_length - full.shape[-1]

        out.append(
            {
                "input_ids": F.pad(full, (0, pad), value=tokenizer.pad_token_id).squeeze(0),
                "labels": F.pad(labels, (0, pad), value=-100).squeeze(0),
                "attention_mask": F.pad(torch.ones_like(full), (0, pad), value=0).squeeze(0),
            }
        )
    return out


def tokenize_jepa_pairs(
    adv_prompts: List[str],
    clean_prompts: List[str],
    responses: List[str],
    intents: List[str],
    tokenizer,
    max_length: int,
) -> List[Dict[str, torch.Tensor]]:
    if not (len(adv_prompts) == len(clean_prompts) == len(responses) == len(intents)):
        raise ValueError("adv_prompts, clean_prompts, responses, and intents must have matching lengths")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out: List[Dict[str, torch.Tensor]] = []
    for adv_prompt, clean_prompt, response, intent in zip(adv_prompts, clean_prompts, responses, intents):
        adv_full = tokenizer.apply_chat_template(
            [{"role": "user", "content": adv_prompt}, {"role": "assistant", "content": response}],
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        adv_full = _chat_template_input_ids(adv_full)[:, :max_length]
        adv_pad = max_length - adv_full.shape[-1]

        clean_prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": clean_prompt}],
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_generation_prompt=True,
        )
        clean_prompt_tokens = _chat_template_input_ids(clean_prompt_tokens)[:, :max_length]
        clean_last_idx = clean_prompt_tokens.shape[-1] - 1
        clean_pad = max_length - clean_prompt_tokens.shape[-1]

        # End-of-prompt index in the adv-side full conversation: where the assistant
        # response starts. Used by jepa_target='prompt_only' to mask out response
        # positions when computing the predictor loss.
        adv_prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": adv_prompt}],
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_generation_prompt=True,
        )
        adv_prompt_end_idx = min(_chat_template_input_ids(adv_prompt_only).shape[-1], adv_full.shape[-1])

        out.append(
            {
                "adv_input_ids": F.pad(adv_full, (0, adv_pad), value=tokenizer.pad_token_id).squeeze(0),
                "adv_attention_mask": F.pad(torch.ones_like(adv_full), (0, adv_pad), value=0).squeeze(0),
                "clean_input_ids": F.pad(clean_prompt_tokens, (0, clean_pad), value=tokenizer.pad_token_id).squeeze(0),
                "clean_attention_mask": F.pad(torch.ones_like(clean_prompt_tokens), (0, clean_pad), value=0).squeeze(0),
                "clean_last_idx": torch.tensor(clean_last_idx, dtype=torch.long),
                "adv_prompt_end_idx": torch.tensor(adv_prompt_end_idx, dtype=torch.long),
                "intent": torch.tensor(1 if intent == "harmful" else 0, dtype=torch.long),
            }
        )
    return out


class TensorDictDataset(Dataset):
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(8192)
        f.seek(0)
        first_non_ws = next((ch for ch in head if not ch.isspace()), None)

        def parse_jsonl(file_obj):
            out = []
            for line in file_obj:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("//"):
                    continue
                out.append(json.loads(s))
            return out

        if first_non_ws == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                f.seek(0)
        return parse_jsonl(f)


def load_ultrachat(num_samples: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=seed).select(range(num_samples))
    prompts, responses = [], []
    for item in ds:
        messages = item["messages"]
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        if user and assistant:
            prompts.append(user)
            responses.append(assistant)
    return prompts, responses


def load_benign_from_path(path: str, num_samples: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Load benign (prompt, response) pairs from a JSON/JSONL file as a drop-in
    replacement for UltraChat. Each row should have at least {prompt, response}
    fields (e.g., from WildJailbreak's vanilla_benign + adversarial_benign rows
    via scripts/build_wildjailbreak_data.py).
    """
    records = _load_json_or_jsonl(path)
    rng = random.Random(seed)
    rng.shuffle(records)
    if num_samples and num_samples < len(records):
        records = records[:num_samples]
    prompts, responses = [], []
    for rec in records:
        prompt = (rec.get("prompt") or rec.get("user") or "").strip()
        response = (rec.get("response") or rec.get("output") or rec.get("assistant") or rec.get("completion") or "").strip()
        if prompt and response:
            prompts.append(prompt)
            responses.append(response)
    return prompts, responses


def load_cb(path: str, limit: int) -> Tuple[List[str], List[str]]:
    records = _load_json_or_jsonl(path)[:limit]
    prompts = [record["prompt"] for record in records]
    harmful_outputs = [record.get("output", "") for record in records]
    return prompts, harmful_outputs


def _load_pair_records(
    pair_path: Optional[str],
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    dataset_split: str,
) -> List[Dict[str, Any]]:
    if pair_path:
        return _load_json_or_jsonl(pair_path)
    if not dataset_name:
        raise ValueError("Either --pair_path or --pair_dataset must be provided.")
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split, delimiter="\t", keep_default_na=False)
    else:
        ds = load_dataset(dataset_name, split=dataset_split)
    return [dict(item) for item in ds]


def _pairs_from_reverse_records(records: List[Dict[str, Any]], limit: Optional[int]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    adv_prompts: List[str] = []
    clean_prompts: List[str] = []
    responses: List[str] = []
    intents: List[str] = []
    sources: List[str] = []
    for rec in records:
        base = rec.get("record", rec)
        clean = _as_text(rec.get("true_prompt") or base.get("prompt") or rec.get("prompt"))
        response = _as_text(rec.get("response") or base.get("output") or rec.get("output"))
        if not clean or not response:
            continue

        original_prompt = _as_text(base.get("prompt") or rec.get("prompt") or rec.get("true_prompt"))
        if original_prompt:
            adv_prompts.append(original_prompt)
            clean_prompts.append(clean)
            responses.append(response)
            intents.append("harmful")
            sources.append("reverse_original")

        for generated in rec.get("generated_prompts") or []:
            adv = _as_text(generated)
            if not adv:
                continue
            adv_prompts.append(adv)
            clean_prompts.append(clean)
            responses.append(response)
            intents.append("harmful")
            sources.append("reverse_generated")
            if limit is not None and len(adv_prompts) >= limit:
                return adv_prompts, clean_prompts, responses, intents, sources

        if limit is not None and len(adv_prompts) >= limit:
            break
    return adv_prompts[:limit], clean_prompts[:limit], responses[:limit], intents[:limit], sources[:limit]


def _pairs_from_wild_records(records: List[Dict[str, Any]], limit: Optional[int]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    adv_prompts: List[str] = []
    clean_prompts: List[str] = []
    responses: List[str] = []
    intents: List[str] = []
    sources: List[str] = []

    if records and "data_type" in records[0] and ("vanilla" in records[0] or "adversarial" in records[0]):
        for rec in records:
            data_type = str(rec.get("data_type", "")).lower()
            if not data_type.startswith("adversarial_"):
                continue
            intent = "benign" if "benign" in data_type else "harmful"
            clean = _as_text(rec.get("vanilla"))
            adv = _as_text(rec.get("adversarial"))
            response = _as_text(rec.get("completion") or rec.get("response") or rec.get("output"))
            if clean and adv and response:
                adv_prompts.append(adv)
                clean_prompts.append(clean)
                responses.append(response)
                intents.append(intent)
                sources.append(f"wildjailbreak_{intent}")
            if limit is not None and len(adv_prompts) >= limit:
                break
        return adv_prompts[:limit], clean_prompts[:limit], responses[:limit], intents[:limit], sources[:limit]

    harmful_clean_keys = (
        "vanilla_harmful",
        "vanilla_harmful_prompt",
        "clean_harmful",
        "clean_harmful_prompt",
    )
    harmful_adv_keys = (
        "adversarial_harmful",
        "adversarial_harmful_prompt",
    )
    harmful_response_keys = (
        "harmful_response",
        "target_response",
        "vanilla_harmful_response",
        "adversarial_harmful_response",
        "response",
        "output",
        "completion",
    )
    benign_clean_keys = (
        "vanilla_benign",
        "vanilla_benign_prompt",
        "clean_benign",
        "clean_benign_prompt",
    )
    benign_adv_keys = (
        "adversarial_benign",
        "adversarial_benign_prompt",
    )
    benign_response_keys = (
        "benign_response",
        "vanilla_benign_response",
        "adversarial_benign_response",
        "response",
        "output",
        "completion",
    )

    for rec in records:
        base = rec.get("record", rec)

        for intent, clean_keys, adv_keys, response_keys in (
            ("harmful", harmful_clean_keys, harmful_adv_keys, harmful_response_keys),
            ("benign", benign_clean_keys, benign_adv_keys, benign_response_keys),
        ):
            clean = _as_text(_first_present(rec, clean_keys) or _first_present(base, clean_keys))
            adv = _as_text(_first_present(rec, adv_keys) or _first_present(base, adv_keys))
            response = _as_text(_first_present(rec, response_keys) or _first_present(base, response_keys))

            if not response:
                response = _as_text(rec.get("y") or base.get("y"))
            if clean and adv and response:
                adv_prompts.append(adv)
                clean_prompts.append(clean)
                responses.append(response)
                intents.append(intent)
                sources.append(f"wildjailbreak_{intent}")

            if limit is not None and len(adv_prompts) >= limit:
                break
        if limit is not None and len(adv_prompts) >= limit:
            break

    return adv_prompts[:limit], clean_prompts[:limit], responses[:limit], intents[:limit], sources[:limit]


def load_jepa_pairs(
    pair_path: Optional[str],
    pair_dataset: Optional[str],
    pair_dataset_config: Optional[str],
    pair_split: str,
    pair_format: str,
    limit: Optional[int],
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    records = _load_pair_records(pair_path, pair_dataset, pair_dataset_config, pair_split)
    if pair_format == "reverse":
        return _pairs_from_reverse_records(records, limit)
    if pair_format == "wildjailbreak":
        return _pairs_from_wild_records(records, limit)
    if pair_format == "auto":
        adv, clean, responses, intents, sources = _pairs_from_wild_records(records, limit)
        if adv:
            return adv, clean, responses, intents, sources
        return _pairs_from_reverse_records(records, limit)
    raise ValueError(f"Unknown pair_format: {pair_format}")


def subsample_parallel_lists(
    lists: Tuple[List[str], List[str], List[str], List[str], List[str]],
    sample_size: int,
    seed: int,
    balanced_by_intent: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    if sample_size <= 0 or len(lists[0]) <= sample_size:
        return lists
    rng = random.Random(seed)
    if balanced_by_intent:
        by_intent: Dict[str, List[int]] = {}
        for idx, intent in enumerate(lists[3]):
            by_intent.setdefault(intent, []).append(idx)
        if len(by_intent) >= 2:
            per_intent = sample_size // len(by_intent)
            indices = []
            leftovers = []
            for intent_indices in by_intent.values():
                rng.shuffle(intent_indices)
                indices.extend(intent_indices[:per_intent])
                leftovers.extend(intent_indices[per_intent:])
            if len(indices) < sample_size:
                indices.extend(rng.sample(leftovers, min(sample_size - len(indices), len(leftovers))))
            rng.shuffle(indices)
            return tuple([[values[i] for i in indices] for values in lists])  # type: ignore[return-value]
    indices = rng.sample(range(len(lists[0])), sample_size)
    return tuple([[values[i] for i in indices] for values in lists])  # type: ignore[return-value]


def concat_parallel_lists(
    left: Tuple[List[str], List[str], List[str], List[str], List[str]],
    right: Tuple[List[str], List[str], List[str], List[str], List[str]],
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    return tuple([a + b for a, b in zip(left, right)])  # type: ignore[return-value]


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def infer_lora_target_modules(model) -> List[str]:
    common_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    seen = set()
    for name, _ in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in common_suffixes:
            seen.add(suffix)
    preferred_order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    resolved = [suffix for suffix in preferred_order if suffix in seen]
    if not resolved:
        raise ValueError("Could not infer LoRA target modules from model.")
    return resolved


@contextmanager
def no_adapter(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


def token_average_ce_per_example(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    denom = valid.sum(dim=-1).clamp(min=1)
    return flat_loss.sum(dim=-1) / denom


def topk_kl_from_logits(adapted_logits: torch.Tensor, base_logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    logp_adapted = F.log_softmax(adapted_logits, dim=-1)
    logp_base = F.log_softmax(base_logits, dim=-1)
    topk_idx = torch.topk(logp_base, k=min(k, logp_base.size(-1)), dim=-1).indices
    adapted_selected = logp_adapted.gather(-1, topk_idx)
    base_selected = logp_base.gather(-1, topk_idx)
    base_probs = base_selected.exp()
    return (base_probs * (base_selected - adapted_selected)).sum(dim=-1).mean()


def parse_int_list(value: Optional[str], default: List[int]) -> List[int]:
    if value is None or value == "":
        return list(default)
    return [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]


def resolve_rep_layers(requested: List[int], hidden_state_count: int) -> List[int]:
    if hidden_state_count <= 1:
        return [0]
    max_index = hidden_state_count - 1
    resolved: List[int] = []
    for raw in requested:
        idx = hidden_state_count + raw if raw < 0 else raw
        idx = min(max(idx, 1), max_index)
        if idx not in resolved:
            resolved.append(idx)
    return resolved or [max_index]


def pooled_assistant_rep(outputs, labels: torch.Tensor, rep_layers: List[int]) -> torch.Tensor:
    hidden_states = outputs.hidden_states
    selected_layers = resolve_rep_layers(rep_layers, len(hidden_states))
    mask = labels.ne(-100).unsqueeze(-1)
    pooled_layers = []
    for layer_idx in selected_layers:
        hidden = hidden_states[layer_idx]
        layer_mask = mask.to(hidden.dtype)
        denom = layer_mask.sum(dim=1).clamp_min(1.0)
        pooled_layers.append((hidden * layer_mask).sum(dim=1) / denom)
    return torch.stack(pooled_layers, dim=0).mean(dim=0)


class HarmRegularizer:
    name = "none"

    def __call__(
        self,
        *,
        model,
        benign_inputs: Dict[str, torch.Tensor],
        harmful_inputs: Dict[str, torch.Tensor],
        benign_out,
        harmful_out,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = model.device
        return torch.zeros((), device=device), {}


class CEFloorRegularizer(HarmRegularizer):
    name = "ce_floor"

    def __init__(self, harm_ce_min: float):
        self.harm_ce_min = harm_ce_min

    def __call__(self, *, model, benign_inputs, harmful_inputs, benign_out, harmful_out):
        harmful_ce = token_average_ce_per_example(harmful_out.logits, harmful_inputs["labels"])
        loss = F.relu(self.harm_ce_min - harmful_ce).pow(2).mean()
        return loss, {
            "loss/harm_ce_floor": loss.detach(),
            "metrics/harm_ce_def": harmful_ce.mean().detach(),
            "metrics/harm_ce_gap_to_floor": (harmful_ce.mean() - self.harm_ce_min).detach(),
        }


class CircuitBreakerRegularizer(HarmRegularizer):
    name = "circuit_breaker"

    def __init__(self, rep_layers: List[int], alpha: float, beta: float,
                 beta_start_mult: float = 1.0, beta_decay_fraction: float = 1.0):
        """beta_start_mult > 1 means the harmful-break term starts (start_mult × beta)
        and decays via cosine to (1.0 × beta) over (beta_decay_fraction × max_steps).
        Default beta_start_mult=1.0 disables the schedule (constant beta)."""
        import math
        self._math = math
        self.rep_layers = rep_layers
        self.alpha = alpha
        self.beta = beta
        self.beta_start_mult = beta_start_mult
        self.beta_decay_fraction = beta_decay_fraction
        self._step = 0
        self._max_steps = 1

    def set_step(self, step: int, max_steps: int) -> None:
        self._step = int(step)
        self._max_steps = max(1, int(max_steps))

    def _current_beta_mult(self) -> float:
        if self.beta_start_mult <= 1.0:
            return 1.0
        denom = max(1.0, self._max_steps * self.beta_decay_fraction)
        progress = min(1.0, self._step / denom)
        # cosine decay: schedule in [0, 1] with schedule(0)=1, schedule(1)=0
        schedule = 0.5 * (1.0 + self._math.cos(self._math.pi * progress))
        # mult in [1.0, beta_start_mult]; starts at beta_start_mult, ends at 1.0
        return 1.0 + (self.beta_start_mult - 1.0) * schedule

    def __call__(self, *, model, benign_inputs, harmful_inputs, benign_out, harmful_out):
        with no_adapter(model), torch.no_grad():
            benign_base_out = model(**benign_inputs, output_hidden_states=True, use_cache=False)
            harmful_base_out = model(**harmful_inputs, output_hidden_states=True, use_cache=False)

        h_b = pooled_assistant_rep(benign_out, benign_inputs["labels"], self.rep_layers)
        h_b_base = pooled_assistant_rep(benign_base_out, benign_inputs["labels"], self.rep_layers)
        h_h = pooled_assistant_rep(harmful_out, harmful_inputs["labels"], self.rep_layers)
        h_h_base = pooled_assistant_rep(harmful_base_out, harmful_inputs["labels"], self.rep_layers)

        benign_anchor = F.mse_loss(h_b.float(), h_b_base.float())
        harmful_break = F.relu(F.cosine_similarity(h_h.float(), h_h_base.float(), dim=-1)).mean()
        beta_mult = self._current_beta_mult()
        loss = self.alpha * benign_anchor + (self.beta * beta_mult) * harmful_break
        return loss, {
            "loss/cb_benign_l2": benign_anchor.detach(),
            "loss/cb_harmful_relu_cos": harmful_break.detach(),
            "loss/cb_beta_mult": torch.tensor(beta_mult),
        }


class TripletRegularizer(HarmRegularizer):
    name = "triplet"

    def __init__(self, rep_layers: List[int], harmful_pull_weight: float, benign_push_weight: float, margin: float):
        self.rep_layers = rep_layers
        self.harmful_pull_weight = harmful_pull_weight
        self.benign_push_weight = benign_push_weight
        self.margin = margin

    def __call__(self, *, model, benign_inputs, harmful_inputs, benign_out, harmful_out):
        h_h = F.normalize(pooled_assistant_rep(harmful_out, harmful_inputs["labels"], self.rep_layers).float(), dim=-1)
        h_b = F.normalize(pooled_assistant_rep(benign_out, benign_inputs["labels"], self.rep_layers).float(), dim=-1)
        harmful_mean = F.normalize(h_h.mean(dim=0, keepdim=True), dim=-1)

        harmful_pull = (1.0 - (h_h * harmful_mean).sum(dim=-1)).mean()
        benign_cos = (h_b * harmful_mean).sum(dim=-1)
        benign_push = F.relu(benign_cos + self.margin).pow(2).mean()
        loss = self.harmful_pull_weight * harmful_pull + self.benign_push_weight * benign_push
        return loss, {
            "loss/triplet_harmful_pull": harmful_pull.detach(),
            "loss/triplet_benign_push": benign_push.detach(),
            "metrics/triplet_benign_cos_to_harm_mean": benign_cos.mean().detach(),
        }


def build_harm_regularizer(args, rep_layers: List[int]) -> HarmRegularizer:
    if args.train_mode == "predictor_only" or args.harm_regularizer == "none":
        return HarmRegularizer()
    if args.harm_regularizer == "ce_floor":
        return CEFloorRegularizer(args.harm_ce_min)
    if args.harm_regularizer == "circuit_breaker":
        return CircuitBreakerRegularizer(
            rep_layers, args.cb_alpha, args.cb_beta,
            beta_start_mult=getattr(args, "cb_beta_start_mult", 1.0),
            beta_decay_fraction=getattr(args, "cb_beta_decay_fraction", 1.0),
        )
    if args.harm_regularizer == "triplet":
        return TripletRegularizer(rep_layers, args.triplet_harmful_pull_weight, args.triplet_benign_push_weight, args.triplet_margin)
    raise ValueError(f"Unknown harm_regularizer: {args.harm_regularizer}")


class PerPositionPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        predictor_type: str = "mlp",
        num_layers: int = 2,
        dropout: float = 0.0,
        bottleneck_dim: int = 0,
    ):
        super().__init__()
        self.predictor_type = predictor_type
        self.bottleneck_dim = bottleneck_dim
        if predictor_type == "identity":
            self.net = nn.Identity()
        elif predictor_type == "linear":
            self.net = nn.Linear(dim, dim)
        elif predictor_type == "mlp":
            if num_layers < 2:
                raise ValueError("--predictor_layers must be >= 2 for predictor_type=mlp")
            if bottleneck_dim > 0:
                layers = [nn.Linear(dim, bottleneck_dim), nn.GELU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                for _ in range(max(0, num_layers - 2)):
                    layers.append(nn.Linear(bottleneck_dim, bottleneck_dim))
                    layers.append(nn.GELU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(bottleneck_dim, dim))
            else:
                layers = []
                for idx in range(num_layers):
                    layers.append(nn.Linear(dim, dim))
                    if idx < num_layers - 1:
                        layers.append(nn.GELU())
                        if dropout > 0:
                            layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoggingCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.csv_path = Path(output_dir) / "metrics.csv"
        self.rows: List[Dict[str, Any]] = []
        self.fieldnames = ["step"]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step}
        row.update(logs)
        for key in row:
            if key not in self.fieldnames:
                self.fieldnames.append(key)
        self.rows.append(row)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for existing in self.rows:
                writer.writerow({k: existing.get(k, "") for k in self.fieldnames})


class TripleBatchLoader:
    def __init__(self, benign_loader: DataLoader, harmful_loader: DataLoader, pair_loader: DataLoader):
        self.benign_loader = benign_loader
        self.harmful_loader = harmful_loader
        self.pair_loader = pair_loader
        self._len = min(len(benign_loader), len(harmful_loader), len(pair_loader))

    def __iter__(self):
        return zip(self.benign_loader, self.harmful_loader, self.pair_loader)

    def __len__(self) -> int:
        return self._len


class JEPACETrainer(Trainer):
    def __init__(
        self,
        benign_ds: Dataset,
        harmful_ds: Dataset,
        pair_ds: Dataset,
        harm_regularizer: HarmRegularizer,
        train_mode: str,
        w_benign: float,
        w_harm: float,
        w_kl: float,
        w_jepa: float,
        jepa_target_encoder: str,
        jepa_target: str,
        predictor_lr_multiplier: float,
        align_layer: int,
        seed: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benign_ds = benign_ds
        self.harmful_ds = harmful_ds
        self.pair_ds = pair_ds
        self.harm_regularizer = harm_regularizer
        self.train_mode = train_mode
        self.w_benign = w_benign
        self.w_harm = w_harm
        self.w_kl = w_kl
        self.w_jepa = w_jepa
        self.jepa_target_encoder = jepa_target_encoder
        self.jepa_target = jepa_target
        self.predictor_lr_multiplier = predictor_lr_multiplier
        self.align_layer = align_layer
        self.seed = seed

    def create_optimizer(self):
        """Override to put predictor params in their own group with predictor_lr_multiplier × base lr.
        Without this the cosine-normalized JEPA loss generates vanishingly small gradients on the
        predictor, leaving its weights essentially at random init while only the LoRA adapts."""
        if self.optimizer is not None:
            return self.optimizer
        from torch.optim import AdamW
        opt_model = self.model
        decay_names = self.get_decay_parameter_names(opt_model)
        named = list(opt_model.named_parameters())
        base_lr = float(self.args.learning_rate)
        wd = float(self.args.weight_decay)
        pred_lr = base_lr * float(self.predictor_lr_multiplier)
        groups = [
            # base LR groups (LoRA + everything except predictor)
            {"params": [p for n, p in named
                        if p.requires_grad and "jepa_predictor" not in n and n in decay_names],
             "weight_decay": wd, "lr": base_lr, "_name": "base_decay"},
            {"params": [p for n, p in named
                        if p.requires_grad and "jepa_predictor" not in n and n not in decay_names],
             "weight_decay": 0.0, "lr": base_lr, "_name": "base_nodecay"},
            # predictor groups (separate higher LR)
            {"params": [p for n, p in named
                        if p.requires_grad and "jepa_predictor" in n and n in decay_names],
             "weight_decay": wd, "lr": pred_lr, "_name": "pred_decay"},
            {"params": [p for n, p in named
                        if p.requires_grad and "jepa_predictor" in n and n not in decay_names],
             "weight_decay": 0.0, "lr": pred_lr, "_name": "pred_nodecay"},
        ]
        groups = [g for g in groups if len(g["params"]) > 0]
        for g in groups:
            n_params = sum(p.numel() for p in g["params"])
            print(f"[optim] group {g.pop('_name')}: {len(g['params'])} tensors ({n_params:,} params), lr={g['lr']:.2e}, wd={g['weight_decay']}")
        optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        # Force AdamW (the cls returned matches that path with default optim setting)
        # Pop lr/wd from optim_kwargs since per-group lr/wd take precedence.
        optim_kwargs.pop("lr", None)
        optim_kwargs.pop("weight_decay", None)
        self.optimizer = optim_cls(groups, **optim_kwargs)
        return self.optimizer

    def get_train_dataloader(self):
        batch_size = self.args.per_device_train_batch_size
        benign_gen = torch.Generator().manual_seed(self.seed)
        harmful_gen = torch.Generator().manual_seed(self.seed + 1)
        pair_gen = torch.Generator().manual_seed(self.seed + 2)
        benign_loader = DataLoader(self.benign_ds, batch_size=batch_size, shuffle=True, generator=benign_gen)
        harmful_loader = DataLoader(self.harmful_ds, batch_size=batch_size, shuffle=True, generator=harmful_gen)
        pair_loader = DataLoader(self.pair_ds, batch_size=batch_size, shuffle=True, generator=pair_gen)
        return TripleBatchLoader(benign_loader, harmful_loader, pair_loader)

    def _resolve_align_layer(self, hidden_states) -> int:
        idx = self.align_layer
        if idx < 0:
            idx = len(hidden_states) + idx
        if idx <= 0 or idx >= len(hidden_states):
            raise ValueError(
                f"align_layer={self.align_layer} resolved to {idx}, but hidden_states has "
                f"{len(hidden_states)} entries. Use an explicit transformer layer index in "
                f"1..{len(hidden_states) - 1}; 0 is the embedding output."
            )
        return idx

    def _jepa_loss(self, model, pair_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = model.device
        adv_ids = pair_batch["adv_input_ids"].to(device)
        adv_mask = pair_batch["adv_attention_mask"].to(device)
        clean_ids = pair_batch["clean_input_ids"].to(device)
        clean_mask = pair_batch["clean_attention_mask"].to(device)
        clean_last_idx = pair_batch["clean_last_idx"].to(device)
        adv_prompt_end_idx = pair_batch.get("adv_prompt_end_idx", None)
        if adv_prompt_end_idx is not None:
            adv_prompt_end_idx = adv_prompt_end_idx.to(device)

        adv_out = model(
            input_ids=adv_ids,
            attention_mask=adv_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        align_layer = self._resolve_align_layer(adv_out.hidden_states)
        adv_hidden = adv_out.hidden_states[align_layer]
        predictor_param = next(model.jepa_predictor.parameters(), None)
        if predictor_param is not None:
            adv_hidden = adv_hidden.to(dtype=predictor_param.dtype)
        pred = model.jepa_predictor(adv_hidden)

        if self.jepa_target_encoder == "base":
            with no_adapter(model), torch.no_grad():
                clean_out = model(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
        elif self.jepa_target_encoder == "defended":
            with torch.no_grad():
                clean_out = model(
                    input_ids=clean_ids,
                    attention_mask=clean_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
        else:
            raise ValueError(f"Unknown jepa_target_encoder: {self.jepa_target_encoder}")
        clean_hidden = clean_out.hidden_states[align_layer]
        batch_idx = torch.arange(clean_hidden.size(0), device=device)
        target = clean_hidden[batch_idx, clean_last_idx, :].detach()

        pred = F.normalize(pred.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)
        sims = (pred * target.unsqueeze(1)).sum(dim=-1)
        token_loss = 1.0 - sims
        mask = adv_mask.to(dtype=token_loss.dtype)
        # If jepa_target=='prompt_only' AND we have the per-row adv prompt end index,
        # restrict the loss to positions BEFORE the assistant response begins.
        if getattr(self, "jepa_target", "with_response") == "prompt_only" and adv_prompt_end_idx is not None:
            B, T = mask.shape
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            prompt_only_mask = (pos < adv_prompt_end_idx.unsqueeze(1)).to(dtype=mask.dtype)
            mask = mask * prompt_only_mask
        return (token_loss * mask).sum() / mask.sum().clamp(min=1.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        benign_batch, harmful_batch, pair_batch = inputs
        device = model.device

        benign_inputs = {k: v.to(device) for k, v in benign_batch.items()}
        harmful_inputs = {k: v.to(device) for k, v in harmful_batch.items()}

        need_hidden = self.w_harm > 0.0 and self.harm_regularizer.name in {"circuit_breaker", "triplet"}
        benign_out = model(**benign_inputs, output_hidden_states=need_hidden, use_cache=False)
        harmful_out = model(**harmful_inputs, output_hidden_states=need_hidden, use_cache=False)

        benign_ce_def = token_average_ce_per_example(benign_out.logits, benign_inputs["labels"])
        harmful_ce_def = token_average_ce_per_example(harmful_out.logits, harmful_inputs["labels"])

        benign_lm_loss = benign_ce_def.mean()
        harmful_lm_loss = harmful_ce_def.mean()

        benign_kl = torch.zeros((), device=device)
        benign_ce_base = torch.zeros((), device=device)
        if self.w_kl > 0.0 and self.train_mode != "predictor_only":
            with no_adapter(model):
                with torch.no_grad():
                    benign_base_out = model(**benign_inputs, use_cache=False)
            benign_ce_base = token_average_ce_per_example(benign_base_out.logits, benign_inputs["labels"]).mean()
            benign_kl = topk_kl_from_logits(benign_out.logits, benign_base_out.logits)

        harm_loss = torch.zeros((), device=device)
        harm_logs: Dict[str, torch.Tensor] = {}
        if self.w_harm > 0.0 and self.train_mode != "predictor_only":
            # If the regularizer supports a step-aware schedule (e.g. circuit_breaker beta decay),
            # tell it the current global step + max_steps before evaluation.
            if hasattr(self.harm_regularizer, "set_step"):
                self.harm_regularizer.set_step(
                    int(self.state.global_step),
                    int(self.args.max_steps if self.args.max_steps and self.args.max_steps > 0 else 1),
                )
            harm_loss, harm_logs = self.harm_regularizer(
                model=model,
                benign_inputs=benign_inputs,
                harmful_inputs=harmful_inputs,
                benign_out=benign_out,
                harmful_out=harmful_out,
            )

        jepa_loss = torch.zeros((), device=device)
        if self.w_jepa > 0.0:
            jepa_loss = self._jepa_loss(model, pair_batch)

        total_loss = (
            (0.0 if self.train_mode == "predictor_only" else self.w_benign) * benign_lm_loss
            + self.w_harm * harm_loss
            + self.w_kl * benign_kl
            + self.w_jepa * jepa_loss
        )

        logs = {
            "loss/lm_benign": benign_lm_loss.item(),
            "loss/lm_harmful": harmful_lm_loss.item(),
            "loss/harm_regularizer": harm_loss.item(),
            "loss/benign_kl": benign_kl.item(),
            "loss/jepa": jepa_loss.item(),
            "loss/total": total_loss.item(),
            "metrics/benign_ce_def": benign_lm_loss.item(),
            "metrics/benign_ce_base": benign_ce_base.item(),
            "metrics/harm_ce_def": harmful_lm_loss.item(),
        }
        logs.update({key: value.item() for key, value in harm_logs.items()})
        self.log(logs)
        return (total_loss, None) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(description="Train JEPA with a pluggable harmful representation regularizer.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cb_path", type=str, required=True)
    parser.add_argument("--init_adapter_path", type=str, default=None)
    parser.add_argument("--train_mode", type=str, default="full", choices=["full", "predictor_only"])
    parser.add_argument("--harm_regularizer", type=str, default="ce_floor", choices=["none", "ce_floor", "circuit_breaker", "triplet"])
    parser.add_argument("--pair_path", type=str, default=None)
    parser.add_argument("--pair_dataset", type=str, default=None)
    parser.add_argument("--pair_dataset_config", type=str, default=None)
    parser.add_argument("--pair_split", type=str, default="train")
    parser.add_argument("--pair_format", type=str, default="auto", choices=["auto", "wildjailbreak", "reverse"])
    parser.add_argument("--extra_pair_path", type=str, default=None)
    parser.add_argument("--extra_pair_dataset", type=str, default=None)
    parser.add_argument("--extra_pair_dataset_config", type=str, default=None)
    parser.add_argument("--extra_pair_split", type=str, default="train")
    parser.add_argument("--extra_pair_format", type=str, default="wildjailbreak", choices=["auto", "wildjailbreak", "reverse"])
    parser.add_argument("--extra_pair_limit", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./jepa_ce")
    parser.add_argument("--ultrachat_samples", type=int, default=DEFAULT_ULTRACHAT_SAMPLES)
    parser.add_argument("--extra_benign_path", type=str, default=None,
                        help="Optional JSON/JSONL with [{prompt, response}, ...] to APPEND to UltraChat "
                             "for the benign side. Used to add WildJailbreak benign rows (vanilla + adv) "
                             "without dropping UltraChat. Pass --ultrachat_samples 0 to use only the extra source.")
    parser.add_argument("--extra_benign_samples", type=int, default=0,
                        help="If >0, cap rows from --extra_benign_path. 0 = use all rows in the file.")
    parser.add_argument("--limit_cb", type=int, default=DEFAULT_CB_LIMIT)
    parser.add_argument("--pair_limit", type=int, default=DEFAULT_PAIR_LIMIT)
    parser.add_argument("--pair_sample_size", type=int, default=0)
    parser.add_argument("--pair_sample_balanced", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--include_pair_harmful_in_harm_ce", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_max_steps", type=int, default=DEFAULT_NUM_MAX_STEPS)
    parser.add_argument("--harm_ce_min", type=float, default=DEFAULT_HARM_CE_MIN)
    parser.add_argument("--w_benign", type=float, default=DEFAULT_W_BENIGN)
    parser.add_argument("--w_harm", type=float, default=DEFAULT_W_HARM)
    parser.add_argument("--w_kl", type=float, default=DEFAULT_W_KL)
    parser.add_argument("--w_jepa", type=float, default=DEFAULT_W_JEPA)
    parser.add_argument("--jepa_target_encoder", type=str, default="base", choices=["base", "defended"])
    parser.add_argument("--jepa_target", type=str, default="with_response",
                        choices=["with_response", "prompt_only"],
                        help="prompt_only: JEPA loss only over adv prompt positions; "
                             "with_response: include adv response positions (current default)")
    parser.add_argument("--predictor_lr_multiplier", type=float, default=1.0,
                        help="Multiply predictor's learning rate by this factor. "
                             "Recommended: 100.0 to compensate for cosine-normalized loss "
                             "shrinking predictor gradients.")
    parser.add_argument("--rep_layers", type=str, default=None)
    parser.add_argument("--cb_alpha", type=float, default=1.0)
    parser.add_argument("--cb_beta", type=float, default=1.0)
    parser.add_argument("--cb_beta_start_mult", type=float, default=1.0,
                        help="At step 0 the harmful-break term is (cb_beta * cb_beta_start_mult). "
                             "Cosine-decays back to cb_beta over (cb_beta_decay_fraction × max_steps). "
                             "Default 1.0 = constant (disabled). Recommended: 10.0 for early-emphasis.")
    parser.add_argument("--cb_beta_decay_fraction", type=float, default=1.0,
                        help="Fraction of max_steps over which the start-mult cosine-decays to 1.0.")
    parser.add_argument("--triplet_harmful_pull_weight", type=float, default=1.0)
    parser.add_argument("--triplet_benign_push_weight", type=float, default=1.0)
    parser.add_argument("--triplet_margin", type=float, default=0.2)
    parser.add_argument("--align_layer", type=int, default=DEFAULT_ALIGN_LAYER)
    parser.add_argument("--predictor_type", type=str, default="mlp", choices=["identity", "linear", "mlp"])
    parser.add_argument("--predictor_layers", type=int, default=2)
    parser.add_argument("--predictor_dropout", type=float, default=0.0)
    parser.add_argument("--predictor_bottleneck_dim", type=int, default=0)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--report_to", type=str, default=DEFAULT_REPORT_TO)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    args = parser.parse_args()
    rep_layers = parse_int_list(args.rep_layers, DEFAULT_REP_LAYERS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    benign_prompts: List[str] = []
    benign_responses: List[str] = []
    if args.ultrachat_samples > 0:
        u_prompts, u_responses = load_ultrachat(args.ultrachat_samples)
        benign_prompts.extend(u_prompts)
        benign_responses.extend(u_responses)
        print(f"[benign] loaded {len(u_prompts)} from UltraChat")
    if args.extra_benign_path:
        e_prompts, e_responses = load_benign_from_path(
            args.extra_benign_path,
            args.extra_benign_samples,  # 0 = all
        )
        benign_prompts.extend(e_prompts)
        benign_responses.extend(e_responses)
        print(f"[benign] appended {len(e_prompts)} from {args.extra_benign_path}")
    print(f"[benign] total: {len(benign_prompts)} prompt/response pairs")
    harmful_prompts, harmful_responses = load_cb(args.cb_path, args.limit_cb)
    pair_adv, pair_clean, pair_responses, pair_intents, pair_sources = load_jepa_pairs(
        pair_path=args.pair_path,
        pair_dataset=args.pair_dataset,
        pair_dataset_config=args.pair_dataset_config,
        pair_split=args.pair_split,
        pair_format=args.pair_format,
        limit=args.pair_limit,
    )
    if args.extra_pair_path or args.extra_pair_dataset:
        extra_pairs = load_jepa_pairs(
            pair_path=args.extra_pair_path,
            pair_dataset=args.extra_pair_dataset,
            pair_dataset_config=args.extra_pair_dataset_config,
            pair_split=args.extra_pair_split,
            pair_format=args.extra_pair_format,
            limit=args.extra_pair_limit or None,
        )
        pair_adv, pair_clean, pair_responses, pair_intents, pair_sources = concat_parallel_lists(
            (pair_adv, pair_clean, pair_responses, pair_intents, pair_sources),
            extra_pairs,
        )
    raw_pair_count = len(pair_adv)
    pair_adv, pair_clean, pair_responses, pair_intents, pair_sources = subsample_parallel_lists(
        (pair_adv, pair_clean, pair_responses, pair_intents, pair_sources),
        args.pair_sample_size,
        args.seed,
        balanced_by_intent=args.pair_sample_balanced,
    )
    if not pair_adv:
        raise ValueError("No JEPA paired views loaded. Check --pair_path/--pair_dataset and --pair_format.")
    print(f"[jepa] pair format={args.pair_format}; loaded {raw_pair_count} pairs; using {len(pair_adv)}")
    intent_counts = {intent: int(sum(x == intent for x in pair_intents)) for intent in sorted(set(pair_intents))}
    source_counts = {source: int(sum(x == source for x in pair_sources)) for source in sorted(set(pair_sources))}
    print(f"[jepa] intent counts={intent_counts}; source counts={source_counts}")
    print(f"[jepa] first pair: intent={pair_intents[0]} adv={pair_adv[0][:80]!r} clean={pair_clean[0][:80]!r}")

    pair_harmful_ce_examples = 0
    if args.include_pair_harmful_in_harm_ce:
        for adv_prompt, response, intent in zip(pair_adv, pair_responses, pair_intents):
            if intent != "harmful":
                continue
            harmful_prompts.append(adv_prompt)
            harmful_responses.append(response)
            pair_harmful_ce_examples += 1
        print(f"[harm_ce] added {pair_harmful_ce_examples} harmful paired adversarial prompts")

    benign_ds = TensorDictDataset(
        tokenize_chat_generic(benign_prompts, benign_responses, tokenizer, max_length=args.max_length)
    )
    harmful_ds = TensorDictDataset(
        tokenize_chat_generic(harmful_prompts, harmful_responses, tokenizer, max_length=args.max_length)
    )
    pair_ds = TensorDictDataset(
        tokenize_jepa_pairs(pair_adv, pair_clean, pair_responses, pair_intents, tokenizer, max_length=args.max_length)
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    target_modules: List[str] = []
    adapter_type = "none"
    if args.init_adapter_path:
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=args.train_mode == "full")
        adapter_type = "lora"
        target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()] if args.target_modules else []
    elif args.train_mode == "full":
        if args.target_modules:
            target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
        else:
            target_modules = infer_lora_target_modules(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        adapter_type = "lora"
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "d_model", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size from model.config.hidden_size or model.config.d_model.")
    hidden_size = int(hidden_size)
    model.jepa_predictor = PerPositionPredictor(
        hidden_size,
        predictor_type=args.predictor_type,
        num_layers=args.predictor_layers,
        dropout=args.predictor_dropout,
        bottleneck_dim=args.predictor_bottleneck_dim,
    )
    model.to(args.device)
    model.jepa_predictor.to(device=args.device, dtype=torch.bfloat16)
    if args.train_mode == "predictor_only":
        for name, param in model.named_parameters():
            if not name.startswith("jepa_predictor."):
                param.requires_grad_(False)
    if args.train_mode == "predictor_only":
        model.eval()
        model.jepa_predictor.train()
    else:
        model.train()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.num_max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_grad_norm=1.0,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
    )

    trainer_api_kwargs = (
        {"processing_class": tokenizer}
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else {"tokenizer": tokenizer}
    )

    trainer = JEPACETrainer(
        model=model,
        benign_ds=benign_ds,
        harmful_ds=harmful_ds,
        pair_ds=pair_ds,
        harm_regularizer=build_harm_regularizer(args, rep_layers),
        train_mode=args.train_mode,
        w_benign=args.w_benign,
        w_harm=args.w_harm,
        w_kl=args.w_kl,
        w_jepa=args.w_jepa,
        jepa_target=args.jepa_target,
        predictor_lr_multiplier=args.predictor_lr_multiplier,
        jepa_target_encoder=args.jepa_target_encoder,
        align_layer=args.align_layer,
        seed=args.seed,
        args=training_args,
        callbacks=[LoggingCallback(args.output_dir)],
        **trainer_api_kwargs,
    )

    trainer.train()
    lora_dir = output_dir / "lora_adapter"
    adapter_path: Optional[str] = None
    if adapter_type == "lora":
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        adapter_path = "lora_adapter"
    else:
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    torch.save(model.jepa_predictor.state_dict(), output_dir / "jepa_predictor.pt")

    manifest = {
        "schema_version": "1.0",
        "defense_name": "jepa_harm_regularized",
        "base_model": args.model,
        "adapter_type": adapter_type,
        "adapter_path": adapter_path,
        "init_adapter_path": args.init_adapter_path,
        "predictor_path": "jepa_predictor.pt",
        "training_completed": True,
        "train_mode": args.train_mode,
        "harm_regularizer": args.harm_regularizer,
        "weights": {
            "w_benign": args.w_benign,
            "w_harm": args.w_harm,
            "w_kl": args.w_kl,
            "w_jepa": args.w_jepa,
        },
        "data": {
            "cb_path": args.cb_path,
            "pair_path": args.pair_path,
            "pair_dataset": args.pair_dataset,
            "pair_dataset_config": args.pair_dataset_config,
            "pair_split": args.pair_split,
            "pair_format": args.pair_format,
            "extra_pair_path": args.extra_pair_path,
            "extra_pair_dataset": args.extra_pair_dataset,
            "extra_pair_dataset_config": args.extra_pair_dataset_config,
            "extra_pair_split": args.extra_pair_split,
            "extra_pair_format": args.extra_pair_format,
            "extra_pair_limit": args.extra_pair_limit,
            "ultrachat_samples": args.ultrachat_samples,
            "limit_cb": args.limit_cb,
            "pair_limit": args.pair_limit,
            "pair_sample_size": args.pair_sample_size,
            "pair_sample_balanced": args.pair_sample_balanced,
            "include_pair_harmful_in_harm_ce": args.include_pair_harmful_in_harm_ce,
            "raw_pair_count": raw_pair_count,
            "num_pair_examples": len(pair_ds),
            "num_harmful_ce_examples": len(harmful_ds),
            "pair_harmful_ce_examples": pair_harmful_ce_examples,
            "intent_counts": intent_counts,
            "source_counts": source_counts,
        },
        "config": {
            "harm_ce_min": args.harm_ce_min,
            "rep_layers": rep_layers,
            "cb_alpha": args.cb_alpha,
            "cb_beta": args.cb_beta,
            "triplet_harmful_pull_weight": args.triplet_harmful_pull_weight,
            "triplet_benign_push_weight": args.triplet_benign_push_weight,
            "triplet_margin": args.triplet_margin,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "align_layer": args.align_layer,
            "jepa_target_encoder": args.jepa_target_encoder,
            "predictor_type": args.predictor_type,
            "predictor_layers": args.predictor_layers,
            "predictor_dropout": args.predictor_dropout,
            "predictor_bottleneck_dim": args.predictor_bottleneck_dim,
            "seed": args.seed,
            "target_modules": target_modules,
        },
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(output_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    (output_dir / "READY").touch()
    print(f"Done. Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
