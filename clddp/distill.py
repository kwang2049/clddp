from __future__ import annotations
from dataclasses import dataclass
from itertools import chain
import logging
import os
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from clddp.dm import (
    JudgedPassage,
    LabeledQuery,
    Split,
)
from clddp.args.distill import RetrievalDistillationArguments
from clddp.retriever import (
    RetrievalTrainingExample,
    Retriever,
    RetrieverConfig,
    SimilarityFunction,
)
from clddp.utils import is_device_zero, set_logger_format, parse_cli
from clddp.dataloader import load_dataset
from clddp.mine import MiningType, load_mined
from clddp.train import RetrievalTrainer, run


@dataclass
class RetrievalDistillationExample(RetrievalTrainingExample):
    scores: List[float]


class RetrieverForDistillation(Retriever):
    def forward(self, examples: List[RetrievalDistillationExample]) -> torch.Tensor:
        if self.config.similarity_function is SimilarityFunction.maxsim:
            raise NotImplementedError("Training ColBERT is yet to be supported")

        queries = [e.query for e in examples]
        num_candidates = len(examples[0].scores)
        passages = list(chain(*(e.passages for e in examples)))
        qembs = self.encode_queries(queries)  # (bsz, hdim)
        pembs, mask = self.encode_passages(passages)  # (bsz * num_candidates, hdim)
        flattened_pembs = pembs.view(
            len(queries), num_candidates, -1
        )  # (bsz, num_candidates, hdim)
        flattened_qembs = qembs.unsqueeze(dim=1).expand_as(flattened_pembs)
        sim_mtrx = (
            self.similarity_function(flattened_qembs, flattened_pembs, pairwise=True)
            * self.config.sim_scale
        )  # (bsz, num_candidates)
        input = torch.nn.functional.log_softmax(sim_mtrx, dim=-1)
        scores = torch.Tensor([e.scores for e in examples]).to(self.device)
        target = torch.softmax(scores, dim=-1)
        loss = torch.nn.functional.kl_div(input, target)
        return {"loss": loss}  # Align with HF's Trainer


class RetrievalDistiller(RetrievalTrainer):
    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor | Any]
    ) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        return loss


class RetrievalDistillationData(Dataset):
    def __init__(self, training_data: List[LabeledQuery], num_candidates: int):
        self.num_candidates = num_candidates
        self.data: List[LabeledQuery] = []
        for lq in training_data:
            jpsgs = lq.get_unique_candidates()
            if len(jpsgs) < num_candidates:
                continue
            for jpsg in jpsgs:
                assert (
                    jpsg.score is not None
                ), f"Score not loaded for query {jpsg.query.query_id} passage {jpsg.passage.passage_id}"
            self.data.append(lq)

    def __getitem__(self, item: int) -> RetrievalDistillationExample:
        lq = self.data[item]
        jpsgs = lq.get_unique_candidates()
        candidates: List[JudgedPassage] = sorted(
            jpsgs, key=lambda jpsg: jpsg.score, reverse=True
        )[: self.num_candidates]
        passages = [jpsg.passage for jpsg in candidates]
        scores = [jpsg.score for jpsg in candidates]
        return RetrievalDistillationExample(
            query=lq.query, passages=passages, scores=scores
        )

    def __len__(self):
        return len(self.data)


def main(args: Optional[RetrievalDistillationArguments] = None) -> Retriever:
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    if args is None:
        args = parse_cli(RetrievalDistillationArguments)

    # Retriever building:
    config = RetrieverConfig(
        query_model_name_or_path=args.query_model_name_or_path,
        passage_model_name_or_path=args.passage_model_name_or_path,
        shared_encoder=args.shared_encoder,
        sep=args.sep,
        pooling=args.pooling,
        similarity_function=args.similarity_function,
        query_max_length=args.query_max_length,
        passage_max_length=args.passage_max_length,
        sim_scale=args.sim_scale,
    )
    retriever = RetrieverForDistillation(config)
    if args.query_prompt is not None:
        retriever.set_query_prompt(args.query_prompt)
    if args.passage_prompt is not None:
        retriever.set_passage_prompt(args.passage_prompt)

    # Data loading:
    train_dataset = load_dataset(
        enable=args.do_train,
        dataloader_name=args.train_dataloader,
        data_name_or_path=args.train_data,
    )
    if args.distillation_path:
        assert args.do_train
        assert train_dataset is not None
        assert args.num_candidates
        load_mined(
            mined_path=args.distillation_path,
            mining_type=MiningType.distillation,
            dataset=train_dataset,
            split=Split.train,
            prograss_bar=is_device_zero(),
        )
    training_data = RetrievalDistillationData(
        training_data=train_dataset.train_labeled_queries,
        num_candidates=args.num_candidates,
    )
    dev_dataset = train_dataset
    if args.dev_data != args.train_data:
        dev_dataset = load_dataset(
            enable=args.do_dev,
            dataloader_name=args.dev_dataloader,
            data_name_or_path=args.dev_data,
        )
    if args.quick_dev:
        assert args.do_dev
        save_pids_to_fpath = (
            os.path.join(args.output_dir, "quick_dev_pids.json")
            if is_device_zero()
            else None
        )
        dev_dataset = dev_dataset.to_quick_version(
            split=Split.dev,
            progress_bar=is_device_zero(),
            save_pids_to_fpath=save_pids_to_fpath,
        )
    test_dataset = train_dataset
    if args.test_data != args.train_data:
        test_dataset = load_dataset(
            enable=args.do_test,
            dataloader_name=args.test_dataloader,
            data_name_or_path=args.test_data,
        )
    if args.quick_test:
        assert args.do_test
        save_pids_to_fpath = (
            os.path.join(args.output_dir, "quick_test_pids.json")
            if is_device_zero()
            else None
        )
        test_dataset = test_dataset.to_quick_version(
            split=Split.test,
            progress_bar=is_device_zero(),
            save_pids_to_fpath=save_pids_to_fpath,
        )

    # Run training:
    run(
        retriever=retriever,
        args=args,
        training_data=training_data,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
    )
    logging.info("Done")
    return Retriever


if __name__ == "__main__":
    # Example cli: torchrun --nproc_per_node=4 --master_port=29501 -m clddp.train
    main()
