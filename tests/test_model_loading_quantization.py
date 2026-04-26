import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BitsAndBytesConfig

_MODEL_LOADING_PATH = Path(__file__).resolve().parents[1] / "adversariallm" / "io_utils" / "model_loading.py"
_SPEC = importlib.util.spec_from_file_location("test_model_loading_module", _MODEL_LOADING_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load model_loading module from {_MODEL_LOADING_PATH}")
model_loading = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model_loading)

build_causallm_load_kwargs = model_loading.build_causallm_load_kwargs
load_model_and_tokenizer = model_loading.load_model_and_tokenizer


def test_build_causallm_load_kwargs_uses_bitsandbytes_for_standard_int4_models():
    kwargs = build_causallm_load_kwargs(
        "meta-llama/Llama-2-7b-chat-hf",
        dtype="int4",
        trust_remote_code=False,
    )

    assert kwargs["device_map"] == "auto"
    assert kwargs["low_cpu_mem_usage"] is True
    assert isinstance(kwargs["quantization_config"], BitsAndBytesConfig)
    assert kwargs["quantization_config"].load_in_4bit is True


@pytest.mark.parametrize("dtype", ["int4", "int8"])
def test_build_causallm_load_kwargs_uses_native_quantization_for_gpt_oss(dtype: str):
    kwargs = build_causallm_load_kwargs(
        "openai/gpt-oss-safeguard-20b",
        dtype=dtype,
        trust_remote_code=False,
    )

    assert kwargs["device_map"] == "auto"
    assert kwargs["low_cpu_mem_usage"] is True
    assert "quantization_config" not in kwargs
    assert "dtype" not in kwargs


def test_build_causallm_load_kwargs_maps_bf16_alias():
    kwargs = build_causallm_load_kwargs(
        "openai/gpt-oss-safeguard-20b",
        dtype="bf16",
        trust_remote_code=False,
    )

    assert kwargs["dtype"] is torch.bfloat16


def test_load_model_and_tokenizer_skips_bitsandbytes_for_gpt_oss_int4(monkeypatch):
    model_calls: list[dict] = []
    tokenizer_calls: list[dict] = []

    class DummyModel:
        def __init__(self):
            self.config = SimpleNamespace()

        def eval(self):
            return self

    def fake_model_from_pretrained(model_id, **kwargs):
        model_calls.append({"model_id": model_id, **kwargs})
        return DummyModel()

    class DummyTokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"
        model_max_length = 4096

    def fake_tokenizer_from_pretrained(model_id, **kwargs):
        tokenizer_calls.append({"model_id": model_id, **kwargs})
        return DummyTokenizer()

    monkeypatch.setattr(
        model_loading.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        model_loading.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )

    model, tokenizer = load_model_and_tokenizer(
        {
            "id": "openai/gpt-oss-safeguard-20b",
            "tokenizer_id": "openai/gpt-oss-safeguard-20b",
            "dtype": "int4",
            "trust_remote_code": False,
            "compile": False,
            "short_name": "gpt_oss",
            "developer_name": "openai",
            "chat_template": None,
        }
    )

    assert model is not None
    assert tokenizer is not None
    assert len(model_calls) == 1
    assert model_calls[0]["model_id"] == "openai/gpt-oss-safeguard-20b"
    assert "quantization_config" not in model_calls[0]
    assert "dtype" not in model_calls[0]
    assert model_calls[0]["device_map"] == "auto"
    assert tokenizer_calls[0]["model_id"] == "openai/gpt-oss-safeguard-20b"
