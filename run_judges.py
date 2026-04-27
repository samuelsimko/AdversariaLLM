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


def _resolve_selected_indices(subrun: dict, cfg: DictConfig) -> set | None:
    selection_cfg = OmegaConf.select(cfg, "selection")
    if selection_cfg is None or selection_cfg.get("top_k") is None:
        return None
    top_k = int(selection_cfg.top_k)
    if top_k <= 0:
        return None
    prescore_classifier = selection_cfg.get("prescore_classifier")
    if not prescore_classifier:
        raise ValueError("selection.prescore_classifier must be set when selection.top_k is enabled")
    prescore_key = build_score_key(prescore_classifier)
    prescore_field = selection_cfg.get("score_field") or default_score_field(prescore_classifier)
    if prescore_field is None:
        raise ValueError(
            f"selection.score_field must be set for prescore classifier '{prescore_classifier}'"
        )
    return set(
        select_top_completion_indices(
            subrun=subrun,
            prescore_key=prescore_key,
            score_field=prescore_field,
            top_k=top_k,
        )
    )


def _extract_subrun_inputs(subrun: dict, score_key: str, cfg: DictConfig):
    """Pull per-subrun prompt lists for batched judging.

    Returns (inputs, metadata) or None if the subrun should be skipped.
    """
    if score_key in subrun["steps"][0]["scores"]:
        return None  # Already scored

    original_conversation = subrun["original_prompt"]
    harmful_prompt = extract_last_user_message(original_conversation)
    selected_indices = _resolve_selected_indices(subrun, cfg)

    without, with_, harmful, attack, response = [], [], [], [], []
    flattened_index = 0
    for step in subrun["steps"]:
        model_input = step["model_input"]
        attack_prompt = extract_last_user_message(model_input)
        for completion in step["model_completions"]:
            if selected_indices is not None and flattened_index not in selected_indices:
                flattened_index += 1
                continue
            without.append(build_without_jailbreak_conversation(original_conversation, model_input, completion))
            with_.append(build_with_jailbreak_conversation(model_input, completion))
            harmful.append(harmful_prompt)
            attack.append(attack_prompt)
            response.append(completion)
            flattened_index += 1

    if not without:
        if selected_indices is not None:
            raise ValueError(
                f"Top-k selection for '{cfg.classifier}' produced no candidates. "
                "Check the prescore judge outputs and ranking field."
            )
        return None

    total_completions = sum(len(step["model_completions"]) for step in subrun["steps"])
    return (
        {"without": without, "with": with_, "harmful": harmful, "attack": attack, "response": response},
        {"selected_indices": selected_indices, "total_completions": total_completions},
    )


def _distribute_subrun_results(subrun: dict, score_key: str, results: dict, metadata: dict) -> int:
    """Slice judge results back into per-step score blocks. Returns count judged."""
    selected_indices = metadata["selected_indices"]
    total_completions = metadata["total_completions"]
    if selected_indices is not None:
        completed = {k: [None] * total_completions for k in results}
        for result_index, completion_index in enumerate(sorted(selected_indices)):
            for key, values in results.items():
                completed[key][completion_index] = values[result_index]
        results = completed

    n_judged = 0
    i = 0
    for step in subrun["steps"]:
        n_completions = len(step["model_completions"])
        step["scores"][score_key] = {k: v[i:i + n_completions] for k, v in results.items()}
        i += n_completions
        n_judged += sum(value is not None for value in next(iter(step["scores"][score_key].values()), []))
    return n_judged


def _process_path_window(
    window_paths: list[str],
    cfg: DictConfig,
    score_key: str,
    judge_holder: list,
    collection,
) -> int:
    """Lock + load every path in the window, judge once, write each back."""
    backend, _ = parse_judge_spec(cfg.classifier)
    loaded = []  # list of (path, lock, attack_run)
    for path in window_paths:
        lock = filelock.FileLock(path + ".lock")
        try:
            lock.acquire(timeout=300)
            with open(path) as f:
                attack_run = json.load(f)
            _, _classifier_name = parse_judge_spec(cfg.classifier)
            _is_overrefusal_classifier = _classifier_name == "overrefusal"
            _is_overrefusal_dataset = attack_run["config"]["dataset"] in ("or_bench", "xs_test")
            if _is_overrefusal_classifier != _is_overrefusal_dataset:
                lock.release()
                continue
            loaded.append((path, lock, attack_run))
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}")
            try:
                lock.release()
            except Exception:
                pass

    if not loaded:
        return 0

    flat_without, flat_with, flat_harmful, flat_attack, flat_response = [], [], [], [], []
    placement = []  # (file_idx, subrun_idx, slot_start, slot_end, metadata)
    for file_idx, (path, _, attack_run) in enumerate(loaded):
        for subrun_idx, subrun in enumerate(attack_run["runs"]):
            try:
                extracted = _extract_subrun_inputs(subrun, score_key, cfg)
            except Exception as e:
                logging.warning(f"{path} subrun {subrun_idx}: {e}")
                continue
            if extracted is None:
                continue
            inputs, metadata = extracted
            slot_start = len(flat_without)
            flat_without.extend(inputs["without"])
            flat_with.extend(inputs["with"])
            flat_harmful.extend(inputs["harmful"])
            flat_attack.extend(inputs["attack"])
            flat_response.extend(inputs["response"])
            slot_end = len(flat_without)
            placement.append((file_idx, subrun_idx, slot_start, slot_end, metadata))

    if not flat_without:
        for _, lock, _ in loaded:
            try:
                lock.release()
            except Exception:
                pass
        return 0

    if judge_holder[0] is None:
        logging.info("Loading judge...")
        judge_holder[0] = build_judge(cfg.classifier)
    judge = judge_holder[0]

    if backend == "judgezoo":
        without_results = judge(flat_without)
        with_results = judge(flat_with)
        big_results = build_judgezoo_dual_results(without_results, with_results)
    else:
        big_results = judge(flat_harmful, flat_attack, flat_response)

    if all(v is None for v in big_results.values()):
        for _, lock, _ in loaded:
            try:
                lock.release()
            except Exception:
                pass
        return 0

    n_judged = 0
    judged_files: set[int] = set()
    for file_idx, subrun_idx, slot_start, slot_end, metadata in placement:
        path, _, attack_run = loaded[file_idx]
        sliced = {k: list(v[slot_start:slot_end]) for k, v in big_results.items()}
        try:
            n_judged += _distribute_subrun_results(
                attack_run["runs"][subrun_idx], score_key, sliced, metadata
            )
            judged_files.add(file_idx)
        except Exception as e:
            logging.warning(f"Failed to write back {path} subrun {subrun_idx}: {e}")

    for file_idx, (path, lock, attack_run) in enumerate(loaded):
        try:
            if file_idx in judged_files:
                with open(path, "w") as f:
                    json.dump(attack_run, f, indent=2, cls=CompactJSONEncoder)
                collection.update_many({"log_file": path}, {"$addToSet": {"scored_by": score_key}})
        except Exception as e:
            logging.warning(f"Failed to save {path}: {e}")
        finally:
            try:
                lock.release()
            except Exception:
                pass

    return n_judged


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

    batch_paths = max(1, int(OmegaConf.select(cfg, "batch_paths", default=1)))
    if batch_paths > 1:
        logging.info(f"Batching {batch_paths} paths per judge call")

    judge_holder: list = [None]
    db = get_mongodb_connection()
    collection = db.runs
    n = 0

    pbar = tqdm(range(0, len(paths), batch_paths), file=sys.stdout)
    for window_start in pbar:
        window = paths[window_start:window_start + batch_paths]
        pbar.set_description(f"{n} judged")
        n += _process_path_window(window, cfg, score_key, judge_holder, collection)
    print(f"Judged {n} completions")


if __name__ == "__main__":
    run_judges()
