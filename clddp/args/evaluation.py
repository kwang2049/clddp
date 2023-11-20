from dataclasses import dataclass
import os
from typing import Optional, Set
from clddp.args.base import AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn
from clddp.dm import Split
from clddp.utils import parse_cli


@dataclass
class EvaluationArguments(AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn):
    checkpoint_dir: Optional[str] = None
    reranker_checkpoint_dir: Optional[str] = None
    data_dir: Optional[str] = None
    dataloader: Optional[str] = None
    evaluation_dir: str = "evaluation"
    split: Split = Split.test
    topk: int = 1000
    per_device_eval_batch_size: int = 32
    fp16: bool = True
    query_prompt: Optional[str] = None
    passage_prompt: Optional[str] = None

    def __post_init__(self) -> None:
        self.split = Split(self.split)
        return super().__post_init__()

    @property
    def escaped_args(self) -> Set[str]:
        return {"evaluation_dir", "dataloader"}

    def build_output_dir(self) -> str:
        return os.path.join(self.evaluation_dir, self.run_name)


if __name__ == "__main__":
    print(parse_cli(EvaluationArguments).output_dir)  # For creating the logging path
