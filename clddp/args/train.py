from dataclasses import dataclass
import os
from typing import Optional, Set
from clddp.args.base import AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn
from clddp.utils import parse_cli
from clddp.retriever import Separator, Pooling, SimilarityFunction
from clddp.dm import Split
from transformers import IntervalStrategy, TrainingArguments


@dataclass
class RetrievalTrainingArguments(
    AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn, TrainingArguments
):
    project: str = "clddp"  # Used for wandb
    checkpoint_dir: str = "checkpoints"

    # Retriever-related:
    query_model_name_or_path: str = (
        "sentence-transformers/msmarco-distilbert-base-tas-b"
    )
    passage_model_name_or_path: Optional[str] = None
    query_prompt: Optional[str] = None
    passage_prompt: Optional[str] = None
    shared_encoder: bool = True
    sep: Separator = Separator.blank
    pooling: Pooling = Pooling.cls
    similarity_function: SimilarityFunction = SimilarityFunction.dot_product
    query_max_length: int = 350
    passage_max_length: int = 350
    sim_scale: float = 1.0
    fp16: bool = True

    # Data-related:
    train_data: Optional[str] = None
    train_dataloader: Optional[str] = None
    num_negatives: int = 0  # 0 means in-batch negatives only
    negatives_path: Optional[str] = None
    positives_path: Optional[str] = None
    dev_data: Optional[str] = None
    dev_dataloader: Optional[str] = None
    test_data: Optional[str] = None
    test_dataloader: Optional[str] = None

    # General training hyperparameters:
    do_train: bool = True
    per_device_train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_train_epochs: int = 5
    warmup_ratio: float = 0.1
    warmup_steps: int = 0

    # Evaluation-related:
    do_dev: bool = True  # whether to evaluate on dev during training
    quick_dev: bool = False  # whether to keep only the labeled passages + sampled passages as the collection for quick dev
    do_test: bool = True
    quick_test: bool = False  # whether to keep only the labeled passages + sampled passages as the collection for quick test
    metric_for_best_model: str = "ndcg_cut_10"
    dev_split: Split = Split.dev  # Note that some datasets do not have a dev split
    test_split: Split = Split.test
    topk: int = 1000
    save_steps: float = 0.1
    eval_steps: float = 0.1  # dev steps
    evaluation_strategy: IntervalStrategy = IntervalStrategy.STEPS

    # Others:
    logging_steps: float = 1
    ddp_find_unused_parameters = False

    @property
    def escaped_args(self) -> Set[str]:
        """Escaped items for building the run_name."""
        return {
            "checkpoint_dir",
            "logging_steps",
            "ddp_find_unused_parameters",
            "save_steps",
            "train_dataloader",
            "dev_dataloader",
            "test_dataloader",
        }

    def __post_init__(self) -> None:
        os.environ[
            "WANDB_PROJECT"
        ] = self.project  # HF's trainer relis on this to know the project name
        self.dev_split = Split(self.dev_split)
        self.test_split = Split(self.test_split)
        self.sep = Separator(self.sep)
        self.pooling = Pooling(self.pooling)
        self.similarity_function = SimilarityFunction(self.similarity_function)
        super().__post_init__()

    def build_output_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(RetrievalTrainingArguments).output_dir
    )  # For creating the logging path
