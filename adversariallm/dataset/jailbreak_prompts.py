"""JailbreakPrompts dataset from HuggingFace.

Source: https://huggingface.co/datasets/simsonsun/JailbreakPrompts

Two splits exist: ``Dataset_1`` and ``Dataset_2``. The default here is
``Dataset_2`` (67 rows). Each row has ``Source`` and ``Prompt`` columns; we
use the ``Prompt`` column verbatim as the user message and pair it with a
generic affirmative target so the existing direct-style attack pipeline
(which assumes a 2-turn ``[user, assistant]`` conversation for loss calc)
works without modification.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

from datasets import load_dataset

from ..types import Conversation
from .prompt_dataset import PromptDataset


@dataclass
class JailbreakPromptsConfig:
    name: str = "jailbreak_prompts"
    hf_dataset: str = "simsonsun/JailbreakPrompts"
    split: str = "Dataset_2"
    seed: int = 0
    idx: Optional[Union[Sequence[int], int, str]] = None
    shuffle: bool = False
    # Placeholder target used only for loss reporting in the direct-style pipeline.
    target: str = "Sure, here is the answer:"


@PromptDataset.register("jailbreak_prompts")
class JailbreakPromptsDataset(PromptDataset):
    def __init__(self, config: JailbreakPromptsConfig):
        super().__init__(config)
        dataset = load_dataset(config.hf_dataset, split=config.split).to_pandas()
        # Drop any blank prompts defensively.
        dataset = dataset[dataset["Prompt"].astype(str).str.strip().astype(bool)].reset_index(drop=True)

        self.idx, self.config_idx = self._select_idx(config, len(dataset))
        selected = dataset.iloc[self.idx.tolist()].reset_index(drop=True)
        self.prompts = selected["Prompt"]
        self.target = config.target

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Conversation:
        return [
            {"role": "user", "content": str(self.prompts.iloc[idx])},
            {"role": "assistant", "content": self.target},
        ]
