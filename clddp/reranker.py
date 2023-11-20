from __future__ import annotations
from dataclasses import asdict, dataclass
import dataclasses
import json
import logging
import os
from typing import Any, Dict, List, Optional, Type, Union
from clddp.dm import Passage, Query, Separator
from clddp.utils import parse_cli, set_logger_format
import torch
from transformers import (
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BertConfig,
    BatchEncoding,
)
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class RerankerConfig:
    model_name_or_path: str
    sep: Separator
    max_length: int

    def __post_init__(self) -> None:
        self.sep = Separator(self.sep)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class RerankerInputExample:
    query: Query
    passages: List[Passage]
    labels: Optional[List[float]] = None


class Reranker(torch.nn.Module):
    """A listwise reranker"""

    CONFIG_FNAME = "reranker_config.json"

    def __init__(self, config: RerankerConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        model_config: BertConfig = AutoConfig.from_pretrained(config.model_name_or_path)
        model_config.num_labels = 1
        self.model: BertForSequenceClassification = (
            AutoModelForSequenceClassification.from_pretrained(
                config.model_name_or_path, config=model_config
            )
        )
        self.set_device("cuda" if torch.cuda.is_available() else "cpu")

    def set_device(self, device: Union[int, str]) -> None:
        self.device = torch.device(device)
        self.to(self.device)

    def predict(self, examples: List[RerankerInputExample]) -> torch.Tensor:
        npassages = len(examples[0].passages)
        input_pairs = [[] for _ in range(npassages)]  # npassages * (batch_size, 2)
        for e in examples:
            for col, passage in enumerate(e.passages):
                input_pairs[col].append(
                    (
                        e.query.text,
                        self.config.sep.concat([passage.title, passage.text]),
                    )
                )
        tokenized: List[BatchEncoding] = [
            self.tokenizer(
                column,
                max_length=self.config.max_length,
                padding=True,
                truncation="only_second",
                return_tensors="pt",
            ).to(self.device)
            for column in input_pairs
        ]  # Each column corresponds to one passage position of the npassages
        predicted: List[SequenceClassifierOutput] = [
            self.model(**column) for column in tokenized
        ]  # npassages * (batch_size, 1)
        scores_list = [
            column_scores.logits.squeeze(dim=-1) for column_scores in predicted
        ]
        scores = torch.stack(scores_list, dim=1)  # (batch_size, npassages)
        return scores

    def forward(self, examples: List[RerankerInputExample]) -> Dict[str, torch.Tensor]:
        assert all(e.labels is not None for e in examples)
        scores = self.predict(examples)
        labels = []  # Positive indices
        for e in examples:
            assert e.labels is not None
            positive_labels = [i for i, label in enumerate(e.labels) if label]
            assert len(positive_labels) == 1
            labels.append(positive_labels[0])
        tensor_labels = torch.LongTensor(labels).to(self.device)
        loss = torch.nn.functional.cross_entropy(scores, tensor_labels)
        return {"loss": loss}

    def save(self, output_dir) -> None:
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
        config_dict = dataclasses.asdict(self.config)
        with open(os.path.join(output_dir, self.CONFIG_FNAME), "w") as f:
            json.dump(config_dict, f, indent=4)
        logging.info(f"Saved to {output_dir}")

    @classmethod
    def from_pretrained(cls: Type[Reranker], checkpoint_dir: str) -> Reranker:
        with open(os.path.join(checkpoint_dir, cls.CONFIG_FNAME)) as f:
            config = RerankerConfig(**json.load(f))
        reranker = cls(config)
        reranker.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir
        )
        reranker.set_device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loaded from {checkpoint_dir}")
        return reranker


@dataclass
class RetrieverBuildingArguments(RerankerConfig):
    output_dir: Optional[str] = None


def main():
    """Build/convert a retriever into the supported format."""
    set_logger_format()
    args = parse_cli(RetrieverBuildingArguments)
    args_dict = dict(vars(args))
    args_dict.pop("output_dir")
    Reranker(RerankerConfig(**args_dict)).save(args.output_dir)
    logging.info(f"Saved reranker checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
