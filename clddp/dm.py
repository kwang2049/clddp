"""Data models."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import json
import logging
import random
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Type

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
class JudgedPassage:
    query: Query
    passage: Passage
    judgement: int
    score: Optional[float] = None


@dataclass
class LabeledQuery:
    query: Query
    positives: List[JudgedPassage]
    negatives: List[JudgedPassage]

    def __post_init__(self):
        assert all(jpsg.judgement > 0 for jpsg in self.positives)
        assert all(jpsg.judgement == 0 for jpsg in self.negatives)
        assert all(
            jpsg.query.query_id == self.query.query_id
            for jpsg in self.positives + self.negatives
        )

    def get_unique_candidates(self) -> List[JudgedPassage]:
        pid2jpsg = {
            jpsg.passage.passage_id: jpsg for jpsg in self.positives + self.negatives
        }
        return list(pid2jpsg.values())

    @staticmethod
    def build_qrels(labeled_queries: List[LabeledQuery]) -> Dict[str, Dict[str, int]]:
        """Build the qrels for trec_eval https://github.com/cvangysel/pytrec_eval."""
        qrels = {}
        for lq in labeled_queries:
            for jpsg in lq.positives:  # Ingore queries without positives
                qrels.setdefault(jpsg.query.query_id, {})
                qrels[jpsg.query.query_id][jpsg.passage.passage_id] = jpsg.judgement
        return qrels

    @staticmethod
    def get_unique_queries(labeled_queries: List[LabeledQuery]) -> List[Query]:
        qid2query = {lq.query.query_id: lq.query for lq in labeled_queries}
        queries = list(qid2query.values())
        return queries

    @staticmethod
    def build_qid2query(labeled_queries: List[LabeledQuery]) -> Dict[str, Query]:
        qid2query = {}
        for lq in labeled_queries:
            qid2query[lq.query.query_id] = lq.query
        return qid2query

    @staticmethod
    def build_qid2positives(
        labeled_queries: List[LabeledQuery],
    ) -> Dict[str, List[JudgedPassage]]:
        qid2positives: Dict[str, List[JudgedPassage]] = {}
        for lq in labeled_queries:
            qid = lq.query.query_id
            for jpsg in lq.positives:
                qid2positives.setdefault(qid, [])
                qid2positives[qid].append(jpsg)
        return qid2positives

    @staticmethod
    def build_qid2negatives(
        labeled_queries: List[LabeledQuery],
    ) -> Dict[str, List[JudgedPassage]]:
        qid2negatives: Dict[str, List[JudgedPassage]] = {}
        for lq in labeled_queries:
            qid = lq.query.query_id
            for jpsg in lq.negatives:
                qid2negatives.setdefault(qid, [])
                qid2negatives[qid].append(jpsg)
        return qid2negatives

    @staticmethod
    def merge(labeled_queries: List[LabeledQuery]) -> List[LabeledQuery]:
        """Merge labeled queries with the same query ID into one."""
        qid2query = {lq.query.query_id: lq.query for lq in labeled_queries}
        qid2positives: Dict[
            str, List[JudgedPassage]
        ] = LabeledQuery.build_qid2positives(labeled_queries)
        qid2negatives: Dict[
            str, List[JudgedPassage]
        ] = LabeledQuery.build_qid2negatives(labeled_queries)
        lqs = []
        for qid, query in qid2query.items():
            lq = LabeledQuery(
                query=query,
                positives=qid2positives.get(qid, []),
                negatives=qid2negatives.get(qid, []),
            )
            lqs.append(lq)
        return lqs


@dataclass
class ScoredPassageID:
    passage_id: str
    score: float

    @staticmethod
    def build_pid2score(scored_passage_ids: List[ScoredPassageID]) -> Dict[str, float]:
        return {spid.passage_id: spid.score for spid in scored_passage_ids}


@dataclass
class RetrievedPassageIDList:
    query_id: str
    scored_passage_ids: List[ScoredPassageID]

    @staticmethod
    def build_trec_scores(
        retrieved: List[RetrievedPassageIDList],
    ) -> Dict[str, Dict[str, float]]:
        trec_scores = {
            rpidl.query_id: ScoredPassageID.build_pid2score(rpidl.scored_passage_ids)
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
        retrieval_results: List[RetrievedPassageIDList],
        fpath: str,
        system: str = "retriever",
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
                        system,
                    )
                    f.write(" ".join(row) + "\n")
        logging.info(f"Dumped retrieval results to {fpath} in TREC format.")

    @classmethod
    def from_trec_csv(
        cls: Type[RetrievedPassageIDList], fpath: str
    ) -> List[RetrievedPassageIDList]:
        logging.info(f"Loading retrieval results from {fpath} in TREC format")
        qid2pid2score: Dict[str, Dict[str, float]] = {}
        with open(fpath, "r") as f:
            for line in f:
                qid, _, pid, rank, score, system = line.strip().split()
                qid2pid2score.setdefault(qid, {})
                qid2pid2score[qid][pid] = float(score)
        loaded: List[cls] = []
        for qid, pid2score in qid2pid2score.items():
            scored: List[ScoredPassageID] = []
            for pid, score in pid2score.items():
                scored.append(ScoredPassageID(passage_id=pid, score=score))
            loaded.append(cls(query_id=qid, scored_passage_ids=scored))
        logging.info("Loaded")
        return loaded


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
    train_labeled_queries: Optional[List[LabeledQuery]] = None
    dev_labeled_queries: Optional[List[LabeledQuery]] = None
    test_labeled_queries: Optional[List[LabeledQuery]] = None

    @property
    def collection_iter(self) -> Iterable[Passage]:
        return self.collection_iter_fn()

    def get_labeled_queries(self, split: Split) -> Optional[List[LabeledQuery]]:
        return {
            Split.train: self.train_labeled_queries,
            Split.dev: self.dev_labeled_queries,
            Split.test: self.test_labeled_queries,
        }[split]

    def set_labeled_queries(
        self, split: Split, labeled_queries: List[LabeledQuery]
    ) -> None:
        if split is Split.train:
            self.train_labeled_queries = labeled_queries
        elif split is Split.dev:
            self.dev_labeled_queries = labeled_queries
        else:
            assert split is Split.test
            self.test_labeled_queries = labeled_queries

    def get_passages(self, pids: List[str], progress_bar: bool) -> List[Passage]:
        pids_set = set(pids)
        pid2passage = {}
        for passage in tqdm.tqdm(
            self.collection_iter,
            desc="Locating the passages",
            total=self.collection_size,
            disable=not progress_bar,
        ):
            if passage.passage_id in pids_set:
                pid2passage[passage.passage_id] = passage
        return [pid2passage[pid] for pid in pids]

    def to_quick_version(
        self,
        split: Split,
        seed: int = 42,
        progress_bar: bool = False,
        save_pids_to_fpath: Optional[str] = None,
    ) -> RetrievalDataset:
        """Build a quick version for the dataset with a trimmed collection containing only the positive passages and a few sampled passages."""
        labeled_queries = self.get_labeled_queries(split)
        random_state = random.Random(seed)
        positives = [jpsg for lq in labeled_queries for jpsg in lq.positives]
        jpsgs_ids = {jpsg.passage.passage_id for jpsg in positives}
        sampled_indices = set(
            random_state.sample(list(range(self.collection_size)), len(jpsgs_ids))
        )
        sampled_passages = []
        for i, psg in enumerate(
            tqdm.tqdm(
                self.collection_iter,
                total=self.collection_size,
                desc=f"Building quick version of the {split} dataset",
                disable=not progress_bar,
            )
        ):
            if i in sampled_indices:
                sampled_passages.append(psg)
        quick_collection = [jpsg.passage for jpsg in positives] + sampled_passages
        pid2psg = {psg.passage_id: psg for psg in quick_collection}
        dedup_quick_collection = list(pid2psg.values())
        random_state.shuffle(dedup_quick_collection)
        quick_dataset = RetrievalDataset(
            collection_iter_fn=lambda: iter(dedup_quick_collection),
            collection_size=len(dedup_quick_collection),
        )
        # Only the specified split is kept:
        quick_dataset.set_labeled_queries(split=split, labeled_queries=labeled_queries)
        if save_pids_to_fpath:
            pids = [psg.passage_id for psg in quick_dataset.collection_iter]
            with open(save_pids_to_fpath, "w") as f:
                json.dump(pids, f)
            logging.info(
                f"Saved passage IDs of the quick {split} dataset to {save_pids_to_fpath} (num.: {len(pids)})"
            )
        return quick_dataset
