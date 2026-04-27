"""Multilingual jailbreak challenges dataset.

Source: https://huggingface.co/datasets/wow2000/multilingual_jailbreak_challenges

Gated (auto-approve) — needs HF_TOKEN or `huggingface-cli login`.

Schema (split=train, ~7K rows):
    jailbreak_prompt_index   (int)   - which jailbreak template
    question_index           (int)   - which underlying harmful question
    en-user_instruction      (str)   - the rendered jailbreak prompt in English
    en-benign_response       (str)   - reference benign answer (NOT a prompt)
    zh-cn-user_instruction, zh-cn-benign_response
    es-user_instruction,    es-benign_response
    ja-user_instruction,    ja-benign_response
    pt-user_instruction,    pt-benign_response
    id-user_instruction,    id-benign_response
    cs-user_instruction,    cs-benign_response
    ar-user_instruction,    ar-benign_response
    fy-user_instruction,    fy-benign_response

We treat every (row, language) pair where the corresponding
``<lang>-user_instruction`` column is non-empty as one independent prompt.
Random subsampling across (jailbreak_prompt_index, question_index, language)
falls out of `shuffle: true` + `idx: "list(range(0, N))"` because rows are
flattened before _select_idx runs.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from datasets import load_dataset

from ..types import Conversation
from .prompt_dataset import PromptDataset


# Suffix that marks a "send this to the model" column.
_PROMPT_SUFFIX = "-user_instruction"


@dataclass
class MultilingualJailbreakConfig:
    name: str = "multilingual_jailbreak"
    hf_dataset: str = "wow2000/multilingual_jailbreak_challenges"
    split: str = "train"
    seed: int = 0
    idx: Optional[Union[Sequence[int], int, str]] = None
    shuffle: bool = False
    # Optional explicit list of language codes (e.g. ["en", "zh-cn", "ar"]). If
    # None, all detected `<lang>-user_instruction` columns are used.
    languages: Optional[Sequence[str]] = None
    prompt_suffix: str = _PROMPT_SUFFIX
    target: str = "Sure, here is the answer:"


@PromptDataset.register("multilingual_jailbreak")
class MultilingualJailbreakDataset(PromptDataset):
    def __init__(self, config: MultilingualJailbreakConfig):
        super().__init__(config)
        dataset = load_dataset(config.hf_dataset, split=config.split).to_pandas()

        suffix = config.prompt_suffix
        prompt_cols: List[str] = [c for c in dataset.columns if isinstance(c, str) and c.endswith(suffix)]
        if not prompt_cols:
            raise ValueError(
                f"No columns ending in '{suffix}' found in {config.hf_dataset} (columns: {list(dataset.columns)}). "
                "Adjust `prompt_suffix` or set `languages` explicitly."
            )

        if config.languages:
            wanted = set(config.languages)
            filtered = [c for c in prompt_cols if c[: -len(suffix)] in wanted]
            missing = wanted - {c[: -len(suffix)] for c in filtered}
            if missing:
                raise ValueError(
                    f"Languages {sorted(missing)} not found among prompt columns "
                    f"{[c[:-len(suffix)] for c in prompt_cols]}"
                )
            prompt_cols = filtered

        # Flatten to (text, language, jailbreak_prompt_index, question_index, row_index) entries.
        entries: List[tuple[str, str, int, int, int]] = []
        for row_idx in range(len(dataset)):
            row = dataset.iloc[row_idx]
            jp_idx = int(row.get("jailbreak_prompt_index", -1))
            q_idx = int(row.get("question_index", -1))
            for col in prompt_cols:
                text = row.get(col)
                if isinstance(text, str) and text.strip():
                    lang = col[: -len(suffix)]
                    entries.append((text, lang, jp_idx, q_idx, row_idx))
        if not entries:
            raise ValueError(
                f"{config.hf_dataset}/{config.split} produced 0 non-empty entries across "
                f"{len(prompt_cols)} prompt columns"
            )

        self.idx, self.config_idx = self._select_idx(config, len(entries))
        selected_idx = self.idx.tolist() if hasattr(self.idx, "tolist") else list(self.idx)
        self.entries = [entries[i] for i in selected_idx]
        self.languages = sorted({e[1] for e in self.entries})
        self.target = config.target

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> Conversation:
        text, _lang, _jp, _q, _row = self.entries[idx]
        return [
            {"role": "user", "content": text},
            {"role": "assistant", "content": self.target},
        ]
