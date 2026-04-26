import os
import json
import random
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"

TEMPLATE = """You are given a model response. Infer a plausible user prompt that could have led to this response.

Response:
{response}

User prompt:
"""

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

@torch.no_grad()
def generate_candidates(
    model,
    tokenizer,
    response_text: str,
    num_return_sequences: int = 10,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
):
    prompt = TEMPLATE.format(response=response_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gens = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

    cleaned = []
    for g in gens:
        g = g.strip()
        if "User prompt:" in g:
            g = g.split("User prompt:", 1)[-1].strip()
        cleaned.append(g)

    return cleaned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--holdout_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="reverse_model_test_generations.json")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--num_generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    holdout = load_json(args.holdout_path)
    if not isinstance(holdout, list):
        raise ValueError("holdout.json should contain a list of (response, prompt) pairs")

    subset = holdout[:args.num_examples]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    results = []
    for i, item in enumerate(subset):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            response_text, true_prompt = item
        elif isinstance(item, dict):
            response_text = item.get("response") or item.get("output") or item.get("y") or ""
            true_prompt = item.get("prompt") or item.get("x") or ""
        else:
            raise ValueError(f"Unsupported holdout item format at index {i}: {type(item)}")

        print(f"[{i+1}/{len(subset)}] Generating candidates...")

        candidates = generate_candidates(
            model=model,
            tokenizer=tokenizer,
            response_text=response_text,
            num_return_sequences=args.num_generations,
        )

        results.append({
            "index": i,
            "response": response_text,
            "true_prompt": true_prompt,
            "generated_prompts": candidates,
        })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_json(results, args.output_path)
    print(f"Saved results to {args.output_path}")

if __name__ == "__main__":
    main()
