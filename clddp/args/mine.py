from dataclasses import dataclass
import os
from typing import Optional, Set
from clddp.args.base import AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn
from clddp.dm import Split
from clddp.utils import parse_cli


@dataclass
class PassageMiningArguments(AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn):
    checkpoint_dir: Optional[str] = None  # The retriever for mining
    data_dir: Optional[str] = None
    dataloader: Optional[str] = None
    mining_dir: str = "mining"
    split: Split = Split.train
    negative_start_ranking: int = 1  # Negative candidates from these rankings
    negative_end_ranking: int = 30  #  Negative candidates to these rankings
    for_distillation: bool = False  # If set, will score all candidates for every query
    positive_start_ranking: int = 1  #  Positive candidates from these rankings
    positive_end_ranking: int = 10  # Positive candidates to these rankings
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_margin_to_negatives: float = 3.0
    default_positive_threshold: Optional[float] = None  # 7.0
    per_device_eval_batch_size: int = 32
    fp16: bool = True
    query_prompt: Optional[str] = None
    passage_prompt: Optional[str] = None

    def __post_init__(self) -> None:
        self.split = Split(self.split)
        assert 0 < self.negative_start_ranking < self.negative_end_ranking
        assert 0 < self.positive_start_ranking < self.positive_end_ranking
        return super().__post_init__()

    @property
    def escaped_args(self) -> Set[str]:
        return {"search_dir", "dataloader"}

    def build_output_dir(self) -> str:
        return os.path.join(self.mining_dir, self.run_name)


if __name__ == "__main__":
    print(parse_cli(PassageMiningArguments).output_dir)  # For creating the logging path
