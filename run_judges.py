import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import json
import logging
import sys

import filelock
import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from adversariallm.errors import print_exceptions
from adversariallm.io_utils import (
    CompactJSONEncoder,
    delete_orphaned_runs,
    get_filtered_and_grouped_paths,
    get_mongodb_connection,
)
from adversariallm.judging import (
    build_judge,
    build_judgezoo_dual_results,
    build_score_key,
    build_with_jailbreak_conversation,
    build_without_jailbreak_conversation,
    default_score_field,
    extract_last_user_message,
    parse_judge_spec,
)

torch.use_deterministic_algorithms(True, warn_only=True)  # determinism
torch.backends.cuda.matmul.allow_tf32 = True


def collect_run_paths(suffixes: list[str]|str, classifier: str, filter_by: dict|None) -> list[str]:
    """
    Collect paths to run files that have not been scored by the specified classifier.

    Args:
        suffixes: List of suffixes that must be in the path
        classifier: Name of the classifier to check in scored_by

    Returns:
        List of paths to run files
    """

    if not isinstance(suffixes, (list, ListConfig)):
        suffixes = [str(suffixes)]
    delete_orphaned_runs()
    db = get_mongodb_connection()
    collection = db.runs

    # Query all stored run metadata from the configured backend.
    all_results = list(collection.find())
    paths = []
    for item in all_results:
        log_file = item["log_file"]
        date_time_string = log_file.split("/")[-3]
        if classifier in item["scored_by"]:
            continue
        if any(date_time_string.endswith(suffix) for suffix in suffixes):
            paths.append(log_file)
    # remove duplicates
    paths = list(set(paths))
    if filter_by:
        filtered_paths = set(get_filtered_and_grouped_paths(OmegaConf.to_container(filter_by, resolve=True))[("all",)])
        paths = [p for p in paths if p in filtered_paths]
    return sorted(paths, reverse=True)


def _coerce_ranking_score(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_top_completion_indices(
    subrun: dict,
    prescore_key: str,
    score_field: str,
    top_k: int,
) -> list[int]:
    scores: list[tuple[float, int]] = []
    offset = 0
    for step in subrun["steps"]:
        completions = step["model_completions"]
        score_block = step["scores"].get(prescore_key)
        if score_block is None:
            raise ValueError(f"Missing prescore judge '{prescore_key}' required for top-k selection")
        values = score_block.get(score_field)
        if values is None:
            raise ValueError(
                f"Missing score field '{score_field}' under prescore judge '{prescore_key}' required for top-k selection"
            )
        if len(values) != len(completions):
            raise ValueError(
                f"Score field '{score_field}' under prescore judge '{prescore_key}' has {len(values)} values, "
                f"expected {len(completions)}"
            )
        for local_index, value in enumerate(values):
            ranking_score = _coerce_ranking_score(value)
            if ranking_score is not None:
                scores.append((ranking_score, offset + local_index))
        offset += len(completions)
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [index for _, index in scores[:top_k]]


@hydra.main(config_path="./conf", config_name="judge", version_base="1.3")
@print_exceptions
def run_judges(cfg: DictConfig) -> None:
    logging.info("-------------------")
    logging.info("Commencing judge run")
    logging.info("-------------------")
    logging.info(cfg)

    score_key = build_score_key(cfg.classifier)
    paths = collect_run_paths(cfg.suffixes, score_key, cfg.filter_by)
    if not paths:
        logging.info("No unjudged paths found")
        return
    logging.info(f"Found {len(paths)} paths")
    logging.info("Loading judge...")
    judge = None
    n = 0
    pbar = tqdm(paths, file=sys.stdout)
    for path in pbar:
        with filelock.FileLock(path + ".lock") as lock:
            try:
                attack_run = json.load(open(path))
                if (cfg.classifier == "overrefusal") != (attack_run["config"]["dataset"] in ("or_bench", "xs_test")):
                    continue
                for subrun in attack_run["runs"]:
                    original_conversation = subrun["original_prompt"]
                    harmful_prompt = extract_last_user_message(original_conversation)
                    without_jailbreak_prompts = []
                    with_jailbreak_prompts = []
                    harmful_prompts = []
                    attack_prompts = []
                    responses = []
                    if score_key in subrun["steps"][0]["scores"]:
                        continue

                    selected_indices = None
                    selection_cfg = OmegaConf.select(cfg, "selection")
                    if selection_cfg is not None and selection_cfg.get("top_k") is not None:
                        top_k = int(selection_cfg.top_k)
                        if top_k > 0:
                            prescore_classifier = selection_cfg.get("prescore_classifier")
                            if not prescore_classifier:
                                raise ValueError("selection.prescore_classifier must be set when selection.top_k is enabled")
                            prescore_key = build_score_key(prescore_classifier)
                            prescore_field = selection_cfg.get("score_field") or default_score_field(prescore_classifier)
                            if prescore_field is None:
                                raise ValueError(
                                    f"selection.score_field must be set for prescore classifier '{prescore_classifier}'"
                                )
                            selected_indices = set(
                                select_top_completion_indices(
                                    subrun=subrun,
                                    prescore_key=prescore_key,
                                    score_field=prescore_field,
                                    top_k=top_k,
                                )
                            )
                    # Late init to avoid loading the judge if not needed
                    if judge is None:
                        judge = build_judge(cfg.classifier)
                    flattened_index = 0
                    for step in subrun["steps"]:
                        model_input = step["model_input"]
                        attack_prompt = extract_last_user_message(model_input)
                        completions: list = step["model_completions"]
                        for completion in completions:
                            if selected_indices is not None and flattened_index not in selected_indices:
                                flattened_index += 1
                                continue
                            without_jailbreak_prompts.append(
                                build_without_jailbreak_conversation(
                                    original_conversation, model_input, completion
                                )
                            )
                            with_jailbreak_prompts.append(
                                build_with_jailbreak_conversation(model_input, completion)
                            )
                            harmful_prompts.append(harmful_prompt)
                            attack_prompts.append(attack_prompt)
                            responses.append(completion)
                            flattened_index += 1
                    pbar.set_description(f"{len(without_jailbreak_prompts)} | {n} total")
                    if not without_jailbreak_prompts:
                        raise ValueError(
                            f"Top-k selection for '{cfg.classifier}' produced no candidates. "
                            "Check the prescore judge outputs and ranking field."
                        )
                    backend, _ = parse_judge_spec(cfg.classifier)
                    if backend == "judgezoo":
                        without_results = judge(without_jailbreak_prompts)
                        with_results = judge(with_jailbreak_prompts)
                        results = build_judgezoo_dual_results(without_results, with_results)
                    else:
                        results = judge(harmful_prompts, attack_prompts, responses)
                    if all(v is None for v in results.values()):
                        continue
                    total_completions = sum(len(step["model_completions"]) for step in subrun["steps"])
                    if selected_indices is not None:
                        completed_results = {k: [None] * total_completions for k in results}
                        selected_order = sorted(selected_indices)
                        for result_index, completion_index in enumerate(selected_order):
                            for key, values in results.items():
                                completed_results[key][completion_index] = values[result_index]
                        results = completed_results
                    i = 0
                    for step in subrun["steps"]:
                        n_completions = len(step["model_completions"])
                        step["scores"][score_key] = {k: v[i:i+n_completions] for k, v in results.items()}
                        i += n_completions
                        n += sum(value is not None for value in next(iter(step["scores"][score_key].values()), []))
                json.dump(attack_run, open(path, "w"), indent=2, cls=CompactJSONEncoder)
                db = get_mongodb_connection()
                collection = db.runs
                collection.update_many({"log_file": path}, {"$addToSet": {"scored_by": score_key}})
            except Exception as e:
                print(path, str(e))
                os.remove(path + ".lock")
                raise Exception(f"Error in {path}. Original exception: {e}") from e
        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")
    print(f"Judged {n} completions")


if __name__ == "__main__":
    run_judges()
