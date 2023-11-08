from abc import ABC, abstractstaticmethod
import csv
import logging
import os
from typing import Dict, List, Optional
from clddp.dm import (
    LabeledQuery,
    JudgedPassage,
    Passage,
    Query,
    RetrievalDataset,
    Split,
)
from clddp.utils import is_device_zero, tqdm_ropen
import ujson


class BaseDataLoader(ABC):
    @abstractstaticmethod
    def load_data(data_name_or_path: str, progress_bar: bool) -> RetrievalDataset:
        pass


class BEIRDataloader(BaseDataLoader):
    @staticmethod
    def load_qrels_from_beir(
        data_dir: str, split: Split
    ) -> Optional[Dict[str, Dict[str, int]]]:
        qrel_path = os.path.join(data_dir, "qrels", f"{split}.tsv")
        if not os.path.exists(qrel_path):
            logging.info(f"Found no {split} split under {data_dir}: {qrel_path}")
            return None

        reader = csv.reader(
            open(qrel_path, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        next(reader)  # skip the header

        qrels = {}
        for row in reader:
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            qrels.setdefault(query_id, {})
            qrels[query_id][corpus_id] = score

        return qrels

    @staticmethod
    def load_data(data_name_or_path: str, progress_bar: bool) -> RetrievalDataset:
        # Loading collection:
        collection: List[Passage] = []
        collection_path = os.path.join(data_name_or_path, "corpus.jsonl")
        for line in tqdm_ropen(
            fpath=collection_path,
            desc=f"Loading collection from {collection_path}",
            pbar=progress_bar,
        ):
            psg_json = ujson.loads(line)
            passage = Passage(passage_id=psg_json["_id"], text=psg_json["text"])
            if psg_json["title"]:
                passage.title = psg_json["title"]
            collection.append(passage)
        pid2psg = {psg.passage_id: psg for psg in collection}

        # Loading queries:
        queries: List[Query] = []
        queries_path = os.path.join(data_name_or_path, "queries.jsonl")
        for line in tqdm_ropen(
            fpath=queries_path,
            desc=f"Loading queries from {queries_path}",
            pbar=progress_bar,
        ):
            query_json = ujson.loads(line)
            queries.append(Query(query_id=query_json["_id"], text=query_json["text"]))
        qid2query = {query.query_id: query for query in queries}

        # Loading qrels:
        split = Split.train
        train_lqs = []
        dev_lqs = []
        test_lqs = []
        for lqs, split in zip([train_lqs, dev_lqs, test_lqs], Split):
            qrels = BEIRDataloader.load_qrels_from_beir(
                data_dir=data_name_or_path, split=split
            )
            if qrels is None:
                continue

            for qid, rels in qrels.items():
                query = qid2query[qid]
                positives = []
                negatives = []
                for pid, rel in rels.items():
                    jpsg = JudgedPassage(
                        query=qid2query[qid], passage=pid2psg[pid], judgement=rel
                    )
                    if rel:
                        positives.append(jpsg)
                    else:
                        negatives.append(jpsg)
                lq = LabeledQuery(query=query, positives=positives, negatives=negatives)
                lqs.append(lq)

        # Return loaded data:
        return RetrievalDataset(
            collection_iter_fn=lambda: iter(collection),
            collection_size=len(collection),
            train_labeled_queries=train_lqs,
            dev_labeled_queries=dev_lqs,
            test_labeled_queries=test_lqs,
        )


DATA_LOADER_LOOKUP: Dict[str, BaseDataLoader] = {"beir": BEIRDataloader}


def load_dataset(
    enable: bool, dataloader_name: Optional[str], data_name_or_path: Optional[str]
) -> Optional[RetrievalDataset]:
    if enable:
        assert dataloader_name is not None
        assert data_name_or_path is not None
        assert dataloader_name in DATA_LOADER_LOOKUP
        return DATA_LOADER_LOOKUP[dataloader_name].load_data(
            data_name_or_path=data_name_or_path, progress_bar=is_device_zero()
        )
    else:
        return None
