"""JailBreakV-28K dataset from HuggingFace.

Source: https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k

We use the ``JailBreakV_28K`` config (28,000 prompts) and the ``jailbreak_query``
column verbatim as the user message. The dataset also exposes a small
``mini_JailBreakV_28K`` split (280 prompts) marked via ``selected_mini=True``;
override ``split`` to use it.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

from datasets import load_dataset

from ..types import Conversation
from .prompt_dataset import PromptDataset


@dataclass
class JailBreakV28KConfig:
    name: str = "jailbreakv_28k"
    hf_dataset: str = "JailbreakV-28K/JailBreakV-28k"
    config_name: str = "JailBreakV_28K"
    split: str = "JailBreakV_28K"
    seed: int = 0
    idx: Optional[Union[Sequence[int], int, str]] = None
    shuffle: bool = False
    # Placeholder target used only for loss reporting in the direct-style pipeline.
    target: str = "Sure, here is the answer:"


@PromptDataset.register("jailbreakv_28k")
class JailBreakV28KDataset(PromptDataset):
    def __init__(self, config: JailBreakV28KConfig):
        super().__init__(config)
        dataset = load_dataset(
            config.hf_dataset,
            config.config_name,
            split=config.split,
        ).to_pandas()
        dataset = dataset[dataset["jailbreak_query"].astype(str).str.strip().astype(bool)].reset_index(drop=True)

        self.idx, self.config_idx = self._select_idx(config, len(dataset))
        selected = dataset.iloc[self.idx.tolist()].reset_index(drop=True)
        self.prompts = selected["jailbreak_query"]
        self.target = config.target

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Conversation:
        return [
            {"role": "user", "content": str(self.prompts.iloc[idx])},
            {"role": "assistant", "content": self.target},
        ]
