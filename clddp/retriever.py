from __future__ import annotations
from dataclasses import asdict, dataclass
import dataclasses
from enum import Enum
from functools import partial
from itertools import chain
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Type, Union
import torch
from sentence_transformers.util import (
    cos_sim,
    dot_score,
    pairwise_cos_sim,
    pairwise_dot_score,
)
from transformers import (
    AutoModelForTextEncoding,
    AutoTokenizer,
    PreTrainedModel,
    BatchEncoding,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch.distributed as dist
from clddp.dm import Passage, Query
from clddp.utils import dist_gather_tensor, colbert_score

MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES[
    "mpnet"
] = "MPNetModel"  # Missing item in the current transformers version


def pairwise_maxsim(
    qembs: torch.Tensor, pembs: torch.Tensor, pmask: torch.BoolTensor
) -> torch.Tensor:
    assert pmask is not None
    return colbert_score(Q=qembs, D_padded=pembs, D_mask=pmask)


def maxsim(
    qembs: torch.Tensor,
    cembs: torch.Tensor,
    cmask: torch.BoolTensor,
    batch_size_query: int = 1024,
    batch_size_passage: int = 16,
) -> torch.Tensor:
    assert cmask is not None
    scores = []
    for bq in range(0, len(qembs), batch_size_query):
        eq = bq + batch_size_query
        qbatch = qembs[bq:eq]
        scores_qbatch = []
        for bc in range(0, len(cembs), batch_size_passage):
            ec = bc + batch_size_passage
            scores_4d = cembs[None, bc:ec] @ qbatch[:, None].to(
                dtype=cembs.dtype
            ).transpose(
                2, 3
            )  # (nqueries, npassages, passage_length, query_length)
            scores_4d += (
                ~cmask[None, bc:ec] * -9999
            )  # (npassages, passage_length, 1) -> (1, npassages, passage_length, 1)
            scores_4d_max: torch.Tensor = scores_4d.max(dim=2)[
                0
            ]  # (nqueries, npassages, query_length)
            scores_2d = scores_4d_max.sum(dim=-1)  # (nqueries, npassages)
            scores_qbatch.append(scores_2d)
        scores.append(torch.cat(scores_qbatch, dim=-1))
    return torch.cat(scores, dim=0)


class Pooling(str, Enum):
    cls = "cls"
    mean = "mean"
    splade = "splade"
    sum = "sum"
    max = "max"
    no_pooling = "no_pooling"

    def __call__(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pooling: (bsz, seq_len, hdim) -> (bsz, hdim) or return the input without pooling."""

        if self == Pooling.cls:
            return token_embeddings[:, 0:1].sum(dim=1)
        elif self == Pooling.mean:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        elif self == Pooling.sum:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1)
        elif self == Pooling.max:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            token_embeddings[
                input_mask_expanded == 0
            ] = -1e9  # Set padding tokens to large negative value
            return torch.max(token_embeddings, 1)[0]
        elif self == Pooling.splade:
            pooled: torch.Tensor = getattr(
                torch.max(
                    torch.log(1 + torch.relu(token_embeddings))
                    * attention_mask.unsqueeze(-1),
                    dim=1,
                ),
                "values",
            )
            return pooled
        elif self == Pooling.no_pooling:
            return token_embeddings
        else:
            return NotImplementedError


class SimilarityFunction(str, Enum):
    """Vector distance between embeddings."""

    dot_product = "dot_product"
    cos_sim = "cos_sim"
    maxsim = "maxsim"

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        passage_embeddings: torch.Tensor,
        passage_mask: Optional[torch.BoolTensor] = None,
        pairwise: bool = False,
    ) -> torch.Tensor:
        """Run the score function over query and passage embeddings."""
        if pairwise:
            # return shape: (npairs,)
            assert len(query_embeddings) == len(passage_embeddings)
            fn = {
                SimilarityFunction.dot_product: pairwise_dot_score,
                SimilarityFunction.cos_sim: pairwise_cos_sim,
                SimilarityFunction.maxsim: partial(pairwise_maxsim, pmask=passage_mask),
            }[self]
        else:
            # resturn shape: (nqueries, npassages)
            fn = {
                SimilarityFunction.dot_product: dot_score,
                SimilarityFunction.cos_sim: cos_sim,
                SimilarityFunction.maxsim: partial(maxsim, pmask=passage_mask),
            }[self]
        scores = fn(query_embeddings, passage_embeddings)
        return scores

    @property
    def __name__(self):
        return {
            SimilarityFunction.dot_product: dot_score,
            SimilarityFunction.cos_sim: cos_sim,
        }[self].__name__


@dataclass
class RetrievalTrainingExample:
    query: Query
    passages: List[Passage]  # (1 positive + n negatives)


class Separator(str, Enum):
    bert_sep = "bert_sep"
    roberta_sep = "roberta_sep"
    blank = "blank"
    empty = "empty"

    @property
    def token(self) -> str:
        return {
            Separator.bert_sep: "[SEP]",
            Separator.roberta_sep: "</s>",
            Separator.blank: " ",
            Separator.empty: "",
        }[self]

    def concat(self, texts: Iterator[Optional[str]]) -> str:
        """Concatenate two pieces of texts with the separation symbol."""
        return self.token.join(filter(lambda text: text is not None, texts))


@dataclass
class RetrieverConfig:
    query_model_name_or_path: str
    shared_encoder: bool
    sep: Separator
    pooling: Pooling
    similarity_function: SimilarityFunction
    max_length: int
    passage_model_name_or_path: Optional[str] = None
    sim_scale: float = 1.0  # For training, used to enlarge gradient

    def __post_init__(self) -> None:
        self.sep = Separator(self.sep)
        self.pooling = Pooling(self.pooling)
        self.similarity_function = SimilarityFunction(self.similarity_function)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict())


class Retriever(torch.nn.Module):
    CONFIG_FNAME = "retriever_config.json"
    SEPARATE_PASSAGE_ENCODER = "separate_passage_encoder"
    PROMPT_SPACE_TOKEN = "[SPACE]"

    _keys_to_ignore_on_save = None  # Required by HF's trainer

    def __init__(self, config: RetrieverConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.query_model_name_or_path)
        self.query_encoder: PreTrainedModel = AutoModelForTextEncoding.from_pretrained(
            config.query_model_name_or_path
        )
        self.passage_encoder = self.query_encoder
        if not config.shared_encoder:
            assert config.passage_model_name_or_path is not None
            self.passage_encoder: PreTrainedModel = (
                AutoModelForTextEncoding.from_pretrained(
                    config.passage_model_name_or_path
                )
            )
        self.sep = Separator(config.sep)
        self.pooling = Pooling(config.pooling)
        self.similarity_function = SimilarityFunction(config.similarity_function)
        self.max_length = config.max_length
        self.set_device("cuda" if torch.cuda.is_available() else "cpu")
        self.query_prompt: Optional[str] = None
        self.passage_prompt: Optional[str] = None

    def set_query_prompt(self, prompt: str) -> None:
        self.query_prompt = prompt.replace(self.PROMPT_SPACE_TOKEN, " ")
        logging.info(f"Set query prompot to {self.query_prompt}")

    def set_passage_prompt(self, prompt: str) -> None:
        self.passage_prompt = prompt.replace(self.PROMPT_SPACE_TOKEN, " ")
        logging.info(f"Set passage prompot to {self.passage_prompt}")

    @staticmethod
    def maybe_add_prompt(prompt: Optional[str], input_text: str) -> str:
        if prompt is not None:
            return prompt + input_text
        else:
            return input_text

    def set_device(self, device: Union[int, str]) -> None:
        self.device = torch.device(device)
        self.to(self.device)

    def encode(
        self, encoder: PreTrainedModel, texts: List[str], batch_size: int
    ) -> torch.Tensor:
        encoded_list: List[torch.Tensor] = []
        for b in range(0, len(texts), batch_size):
            e = b + batch_size
            tokenized: BatchEncoding = self.tokenizer(
                texts[b:e],
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            outputs: BaseModelOutputWithPoolingAndCrossAttentions = encoder(**tokenized)
            pooled = self.pooling(
                token_embeddings=outputs.last_hidden_state,
                attention_mask=tokenized["attention_mask"],
            )  # (bsz, hdim)
            encoded_list.append(pooled)
        return torch.cat(encoded_list)

    def encode_queries(
        self, queries: List[Query], batch_size: int = 16
    ) -> torch.Tensor:
        return self.encode(
            encoder=self.query_encoder,
            texts=[
                self.maybe_add_prompt(prompt=self.query_prompt, input_text=query.text)
                for query in queries
            ],
            batch_size=batch_size,
        )

    def encode_passages(
        self, passages: List[Passage], batch_size: int = 16
    ) -> torch.Tensor:
        return self.encode(
            encoder=self.passage_encoder,
            texts=[
                self.maybe_add_prompt(
                    prompt=self.passage_prompt,
                    input_text=self.sep.concat([passage.title, passage.text]),
                )
                for passage in passages
            ],
            batch_size=batch_size,
        )

    def forward(self, examples: List[RetrievalTrainingExample]) -> torch.Tensor:
        queries = [e.query for e in examples]
        passages = list(chain(*(e.passages for e in examples)))
        qembs = self.encode_queries(queries)  # (bsz, hdim)
        pembs = self.encode_passages(passages)  # (bsz * (1 + nnegs), hdim)
        if dist.is_initialized():
            # Key in multi-gpu training for contrastive learning:
            # https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
            qembs = dist_gather_tensor(qembs)
            pembs = dist_gather_tensor(pembs)
        sim_mtrx = self.similarity_function(qembs, pembs) * self.config.sim_scale
        npsgs_per_query = len(pembs) // len(qembs)
        labels = (
            torch.arange(sim_mtrx.size(0), device=sim_mtrx.device, dtype=torch.long)
            * npsgs_per_query
        )
        loss = torch.nn.functional.cross_entropy(sim_mtrx, labels)
        if dist.is_initialized():
            loss *= dist.get_world_size()
        return {"loss": loss}  # Align with HF's Trainer

    def save(self, output_dir) -> None:
        self.tokenizer.save_pretrained(output_dir)
        self.query_encoder.save_pretrained(output_dir)
        if not self.config.shared_encoder:
            self.passage_encoder.save_pretrained(
                os.path.join(output_dir, self.SEPARATE_PASSAGE_ENCODER)
            )
        config_dict = dataclasses.asdict(self.config)
        with open(os.path.join(output_dir, self.CONFIG_FNAME), "w") as f:
            json.dump(config_dict, f, indent=4)
        logging.info(f"Saved to {output_dir}")

    @classmethod
    def from_pretrained(cls: Type[Retriever], checkpoint_dir: str) -> Retriever:
        with open(os.path.join(checkpoint_dir, cls.CONFIG_FNAME)) as f:
            config = RetrieverConfig(**json.load(f))
        retriever = cls(config)
        retriever.query_encoder = AutoModelForTextEncoding.from_pretrained(
            checkpoint_dir
        )
        retriever.passage_encoder = retriever.query_encoder
        if not config.shared_encoder:
            retriever.passage_encoder = AutoModelForTextEncoding.from_pretrained(
                os.path.join(checkpoint_dir, cls.SEPARATE_PASSAGE_ENCODER)
            )
        retriever.set_device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loaded from {checkpoint_dir}")
        return retriever


@dataclass
class RetrieverBuildingArguments(RetrieverConfig):
    output_dir: Optional[str] = None


if __name__ == "__main__":
    # Build/convert a retriever into the supported format
    from clddp.utils import parse_cli, set_logger_format

    set_logger_format()
    args = parse_cli(RetrieverBuildingArguments)
    args_dict = dict(vars(args))
    args_dict.pop("output_dir")
    Retriever(RetrieverConfig(**args_dict)).save(args.output_dir)
    logging.info(f"Saved retriever checkpoint to {args.output_dir}")
