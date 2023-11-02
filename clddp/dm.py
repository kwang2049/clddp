"""Data models."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Callable, Dict, Iterable, List, Optional, Type

import tqdm


@dataclass
class Passage:
    passage_id: str
    text: str
    title: Optional[str] = None


@dataclass
class Query:
    query_id: str
    text: str


@dataclass
class JudgedPassage:
    query: Query
    passage: Passage
    judgement: int

    @staticmethod
    def build_qrels(judeged_passages: List[JudgedPassage]) -> Dict[str, Dict[str, int]]:
        """Build the qrels for trec_eval https://github.com/cvangysel/pytrec_eval."""
        qrels = {}
        for jp in judeged_passages:
            qrels.setdefault(jp.query.query_id, {})
            qrels[jp.query.query_id][jp.passage.passage_id] = jp.judgement
        return qrels

    @staticmethod
    def get_unique_queries(judged_passages: List[JudgedPassage]) -> List[Query]:
        qid2query = {jpsg.query.query_id: jpsg.query for jpsg in judged_passages}
        queries = list(qid2query.values())
        return queries


@dataclass
class ScoredPassageID:
    passage_id: str
    score: float


@dataclass
class RetrievedPassageIDList:
    query_id: str
    scored_passage_ids: List[ScoredPassageID]

    @staticmethod
    def build_trec_scores(
        retrieved: List[RetrievedPassageIDList],
    ) -> Dict[str, Dict[str, float]]:
        trec_scores = {
            rpidl.query_id: {
                spid.passage_id: spid.score for spid in rpidl.scored_passage_ids
            }
            for rpidl in retrieved
        }
        return trec_scores

    @classmethod
    def merge(
        cls: Type[RetrievedPassageIDList],
        rpidls: List[RetrievedPassageIDList],
        topk: Optional[int] = None,
    ) -> RetrievedPassageIDList:
        qids = list(set(rpidl.query_id for rpidl in rpidls))
        assert len(qids) == 1, "Can only merge results for the same query"
        pid2score = {
            spid.passage_id: spid.score
            for rpidl in rpidls
            for spid in rpidl.scored_passage_ids
        }
        if topk:
            pid2score = dict(
                sorted(
                    pid2score.items(), key=lambda pid_score: pid_score[1], reverse=True
                )[:topk]
            )
        return RetrievedPassageIDList(
            query_id=qids[0],
            scored_passage_ids=[
                ScoredPassageID(passage_id=pid, score=score)
                for pid, score in pid2score.items()
            ],
        )

    def dump_trec_csv(
        retrieval_results: List[RetrievedPassageIDList], fpath: str
    ) -> None:
        with open(fpath, "w") as f:
            for ranking in tqdm.tqdm(
                retrieval_results, desc="Saving retrieval results"
            ):
                sorted_spsg_ids = sorted(
                    ranking.scored_passage_ids,
                    key=lambda spsg_id: spsg_id.score,
                    reverse=True,
                )
                for i, spsg_id in enumerate(sorted_spsg_ids):
                    row = (
                        ranking.query_id,
                        "Q0",
                        spsg_id.passage_id,
                        str(i + 1),
                        str(spsg_id.score),
                        "clddp",
                    )
                    f.write(" ".join(row) + "\n")
        logging.info(f"Dumped retrieval results to {fpath} in TREC format.")


class Split(str, Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class RetrievalDataset:
    collection_iter_fn: Callable[
        [], Iterable[Passage]
    ]  # This design is for handling very large collections
    collection_size: int
    judged_passages_train: Optional[List[JudgedPassage]] = None
    judged_passages_dev: Optional[List[JudgedPassage]] = None
    judged_passages_test: Optional[List[JudgedPassage]] = None

    @property
    def collection_iter(self) -> Iterable[Passage]:
        return self.collection_iter_fn()

    def get_judged_passages(self, split: Split) -> Optional[List[JudgedPassage]]:
        return {
            Split.train: self.judged_passages_train,
            Split.dev: self.judged_passages_dev,
            Split.test: self.judged_passages_test,
        }[split]
