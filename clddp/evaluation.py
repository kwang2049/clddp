from enum import Enum
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
from clddp.reranker import Reranker, RerankerInputExample
import numpy as np
import pytrec_eval
from clddp.dm import (
    LabeledQuery,
    Passage,
    RetrievalDataset,
    RetrievedPassageIDList,
    ScoredPassageID,
    Split,
)
from clddp.dataloader import load_dataset
from clddp.retriever import Retriever
from clddp.search import rerank, search
import torch.distributed as dist
from clddp.utils import (
    get_rank,
    is_device_zero,
    set_logger_format,
    parse_cli,
    initialize_ddp,
)
from clddp.args.evaluation import EvaluationArguments
from transformers import HfArgumentParser
import tqdm
import torch


class RetrievalMetric(str, Enum):
    map_string = "map_cut"
    ndcg_string = "ndcg_cut"
    recall_string = "recall"
    precision_string = "P"
    rr_string = "recip_rank"

    @staticmethod
    def cutoff(metric_string: str) -> int:
        return int(metric_string.split("_")[-1])

    def at(self, k: int) -> str:
        return f"{self}_{k}"

    def trec_string(self, k_values: Tuple[int]) -> str:
        if self is RetrievalMetric.rr_string:
            return self

        return f"{self}." + ",".join([str(k) for k in k_values])


class RetrievalEvaluator:
    def __init__(
        self,
        eval_dataset: RetrievalDataset,
        split: Split,
        metrics: Tuple[str] = (
            RetrievalMetric.ndcg_string.at(10),
            RetrievalMetric.rr_string.at(10),
            RetrievalMetric.recall_string.at(100),
        ),
        precision: int = 4,
    ) -> None:
        self.labeled_queries = eval_dataset.get_labeled_queries(split)
        self.queries = LabeledQuery.get_unique_queries(self.labeled_queries)
        self.qrels = LabeledQuery.build_qrels(self.labeled_queries)
        self.metrics = metrics
        self.precision = precision

    def __call__(
        self, retrieved_passage_id_lists: List[RetrievedPassageIDList]
    ) -> Dict[str, float]:
        trec_scores = RetrievedPassageIDList.build_trec_scores(
            retrieved_passage_id_lists
        )
        eval_scores = self._pytrec_eval(trec_scores=trec_scores, qrels=self.qrels)
        report = self._build_report(eval_scores=eval_scores)
        return report

    def _post_process_rr(self, eval_scores: Dict[str, Dict[str, float]]) -> None:
        """pytrec_eval does not support MRR at K originally."""
        rr_cutoffs = [
            int(m.split("_")[-1])
            for m in self.metrics
            if RetrievalMetric.rr_string in m
        ]
        for k in rr_cutoffs:
            min_rr = 1 / k
            for metric2score in eval_scores.values():
                score = metric2score[RetrievalMetric.rr_string]
                if score < min_rr:
                    score = 0
                metric2score[RetrievalMetric.rr_string.at(k)] = score

    def _pytrec_eval(
        self, trec_scores: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        logging.info("Running pytrec-eval")
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, self.metrics)
        eval_scores: Dict[str, Dict[str, float]] = evaluator.evaluate(trec_scores)
        self._post_process_rr(eval_scores)
        return eval_scores

    def _build_report(
        self, eval_scores: Dict[str, Dict[str, float]], prefix: str = ""
    ) -> Dict[str, float]:
        report: Dict[str, float] = {}
        for metric in self.metrics:
            item_name = f"{prefix}{metric}"
            report[item_name] = round(
                np.mean([m2s[metric] for m2s in eval_scores.values()]).tolist(),
                self.precision,
            )
        report = dict(sorted(report.items(), key=lambda kv: kv[0]))
        return report


def search_and_evaluate(
    retriever: Retriever,
    eval_dataset: RetrievalDataset,
    split: Split,
    topk: int,
    per_device_eval_batch_size: int,
    fp16: bool,
    metric_key_prefix: str = "eval",
) -> Tuple[List[RetrievedPassageIDList], Dict[str, float]]:
    # Build evaluator and do search:
    evaluator = RetrievalEvaluator(eval_dataset=eval_dataset, split=split)
    rpidls = search(
        retriever=retriever,
        collection_iter=eval_dataset.collection_iter,
        collection_size=eval_dataset.collection_size,
        queries=evaluator.queries,
        topk=topk,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=fp16,
    )

    # Calculate scores:
    report = evaluator(rpidls)
    report_prefixed = {}
    for k, v in report.items():
        report_prefixed[f"{metric_key_prefix}_{k}"] = v  # Required by HF's trainer

    if dist.is_initialized():
        dist.barrier()

    logging.info(f"Evaluation results: {report_prefixed}")
    return rpidls, report_prefixed


def rerank_and_evaluate(
    retrieval_results: List[RetrievedPassageIDList],
    reranker: Reranker,
    eval_dataset: RetrievalDataset,
    split: Split,
    per_device_eval_batch_size: int,
    fp16: bool,
    metric_key_prefix: str = "eval",
) -> Tuple[List[RetrievedPassageIDList], Dict[str, float]]:
    # Build evaluator and do search:
    queries = LabeledQuery.get_unique_queries(eval_dataset.get_labeled_queries(split))
    reranked = rerank(
        retrieval_results=retrieval_results,
        reranker=reranker,
        collection_iter=eval_dataset.collection_iter,
        collection_size=eval_dataset.collection_size,
        queries=queries,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=fp16,
    )

    # Calculate scores:
    evaluator = RetrievalEvaluator(eval_dataset=eval_dataset, split=split)
    report = evaluator(reranked)
    report_prefixed = {}
    for k, v in report.items():
        report_prefixed[f"{metric_key_prefix}_{k}"] = v  # Required by HF's trainer

    if dist.is_initialized():
        dist.barrier()

    logging.info(f"Evaluation results: {report_prefixed}")
    return reranked, report_prefixed


def main(
    args: Optional[EvaluationArguments] = None,
) -> Union[List[RetrievedPassageIDList], Dict[str, float]]:
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    args = parse_cli(EvaluationArguments)
    if is_device_zero():
        args.dump_arguments()

    retriever = Retriever.from_pretrained(args.checkpoint_dir)
    if args.query_prompt is not None:
        retriever.set_query_prompt(args.query_prompt)
    if args.passage_prompt is not None:
        retriever.set_passage_prompt(args.passage_prompt)
    retriever.set_device(get_rank())
    eval_dataset = load_dataset(
        enable=True, dataloader_name=args.dataloader, data_name_or_path=args.data_dir
    )
    ranking_results, report = search_and_evaluate(
        retriever=retriever,
        eval_dataset=eval_dataset,
        split=args.split,
        topk=args.topk,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
    )
    system = "retriever"
    if args.reranker_checkpoint_dir:
        reranker = Reranker.from_pretrained(args.reranker_checkpoint_dir)
        ranking_results, report = rerank_and_evaluate(
            retrieval_results=ranking_results,
            reranker=reranker,
            eval_dataset=eval_dataset,
            split=args.split,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.fp16,
        )
        system = "retriever+reranker"
    franked = os.path.join(args.output_dir, "ranking_results.txt")
    if is_device_zero():
        RetrievedPassageIDList.dump_trec_csv(
            retrieval_results=ranking_results, fpath=franked, system=system
        )
        freport = os.path.join(args.output_dir, "metrics.json")
        with open(freport, "w") as f:
            json.dump(report, f, indent=4)
        logging.info(f"Saved evaluation metrics to {freport}.")
    logging.info("Done")
    return ranking_results, report


if __name__ == "__main__":
    main()
