"""
@article{cui2024or,
  title={Or-bench: An over-refusal benchmark for large language models},
  author={Cui, Justin and Chiang, Wei-Lin and Stoica, Ion and Hsieh, Cho-Jui},
  journal={arXiv preprint arXiv:2405.20947},
  year={2024}
}
"""
from dataclasses import dataclass

from datasets import load_dataset

from ..types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class ORBenchConfig:
    name: str = "or_bench"
    config_name: str = "or-bench-hard-1k"  # other options: "or-bench-80k", "or-bench-toxic"
    split: str = "train"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("or_bench")
class ORBenchDataset(PromptDataset):
    def __init__(self, config: ORBenchConfig):
        super().__init__(config)
        dataset = load_dataset("bench-llm/or-bench", config.config_name)[config.split]
        dataset = [d["prompt"] for d in dataset]

        self.idx, self.config_idx = self._select_idx(config, len(dataset))

        self.messages = [dataset[i] for i in self.idx]
        self.targets = [""] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.messages[idx]
        conversation = [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": ""},
        ]
        return conversation
