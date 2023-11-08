"""Mine hard negatives."""
import logging
import os
from typing import Dict, List, Optional
import numpy as np

import tqdm
from clddp.args.mine import NegativeMiningArguments
from clddp.search import search
from clddp.dm import (
    LabeledQuery,
    Passage,
    JudgedPassage,
    Query,
    RetrievedPassageIDList,
    ScoredPassageID,
    RetrievalDataset,
    Split,
)
from clddp.dataloader import load_dataset
from clddp.retriever import Retriever
from sentence_transformers.cross_encoder import CrossEncoder
from clddp.utils import (
    get_rank,
    initialize_ddp,
    is_device_zero,
    parse_cli,
    set_logger_format,
    split_data,
    split_data_size,
    all_gather_object,
)


def keep_range(
    retrieval_results: List[RetrievedPassageIDList],
    start_ranking: int,
    end_ranking: int,
) -> List[RetrievedPassageIDList]:
    kept_rrs = []
    for rr in retrieval_results:
        kept = sorted(rr.scored_passage_ids, key=lambda spid: spid.score, reverse=True)[
            start_ranking - 1 : end_ranking
        ]
        kept_rrs.append(
            RetrievedPassageIDList(query_id=rr.query_id, scored_passage_ids=kept)
        )
    return kept_rrs


def score_query_passages(
    cross_encoder: CrossEncoder, query: Query, passages: List[Passage]
) -> List[float]:
    pairs = [
        (
            query.text,
            psg.text
            if psg.title is None
            else cross_encoder.tokenizer.sep_token.join([psg.title, psg.text]),
        )
        for psg in passages
    ]
    scores = cross_encoder.predict(pairs, show_progress_bar=False).tolist()
    return scores


def main(args: Optional[NegativeMiningArguments] = None):
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    if args is None:
        args = parse_cli(NegativeMiningArguments)
    if is_device_zero():
        args.dump_arguments()

    # Doing search:
    retriever = Retriever.from_pretrained(args.checkpoint_dir)
    if args.query_prompt is not None:
        retriever.set_query_prompt(args.query_prompt)
    if args.passage_prompt is not None:
        retriever.set_passage_prompt(args.passage_prompt)
    retriever.set_device(get_rank())
    dataset = load_dataset(
        enable=True, dataloader_name=args.dataloader, data_name_or_path=args.data_dir
    )
    labeled_queries = LabeledQuery.merge(dataset.get_labeled_queries(args.split))
    queries = LabeledQuery.get_unique_queries(labeled_queries)
    retrieval_results: List[RetrievedPassageIDList] = search(
        retriever=retriever,
        collection_iter=dataset.collection_iter,
        collection_size=dataset.collection_size,
        queries=queries,
        topk=args.end_ranking,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
    )

    # Keep candidates in the specified range:
    kept_retrieval_results = keep_range(
        retrieval_results=retrieval_results,
        start_ranking=args.start_ranking,
        end_ranking=args.end_ranking,
    )
    if is_device_zero():
        RetrievedPassageIDList.dump_trec_csv(
            retrieval_results=kept_retrieval_results,
            fpath=os.path.join(args.output_dir, "mined_without_filtering.txt"),
            system="retriever",
        )

    # Do filtering with the cross-encoder:
    pids = {
        spid.passage_id
        for rr in kept_retrieval_results
        for spid in rr.scored_passage_ids
    }
    pid2psg: Dict[str, Passage] = {}
    for psg in tqdm.tqdm(
        dataset.collection_iter,
        total=dataset.collection_size,
        desc="Building pid2psg",
        disable=not is_device_zero(),
    ):
        if psg.passage_id not in pids:
            continue
        pid2psg[psg.passage_id] = psg
    qid2positives = LabeledQuery.build_qid2positives(labeled_queries)
    qid2query = LabeledQuery.build_qid2query(labeled_queries)
    cross_encoder = CrossEncoder(args.cross_encoder)
    mined_negatives: List[RetrievedPassageIDList] = []
    for rr in tqdm.tqdm(
        split_data(kept_retrieval_results),
        desc="Filtering true negatives",
        total=split_data_size(len(kept_retrieval_results)),
        disable=not is_device_zero(),
    ):
        query = qid2query[rr.query_id]
        positives = [jpsg.passage for jpsg in qid2positives[rr.query_id]]
        positive_scores = score_query_passages(
            cross_encoder=cross_encoder, query=query, passages=positives
        )
        avg_positive_score = np.mean(positive_scores)
        positive_ids = {psg.passage_id for psg in positives}
        negatives = [
            pid2psg[spid.passage_id]
            for spid in rr.scored_passage_ids
            if spid.passage_id not in positive_ids
        ]
        negative_scores = score_query_passages(
            cross_encoder=cross_encoder, query=query, passages=negatives
        )
        scored_passage_ids = []
        for psg, score in zip(negatives, negative_scores):
            if avg_positive_score - score > args.cross_encoder_margin:
                scored_passage_ids.append(
                    ScoredPassageID(passage_id=psg.passage_id, score=score)
                )
        mined_negatives.append(
            RetrievedPassageIDList(
                query_id=query.query_id, scored_passage_ids=scored_passage_ids
            )
        )
    gathered_mined_negatives = sum(all_gather_object(mined_negatives), [])
    if is_device_zero():
        fretrieved = os.path.join(args.output_dir, "mined_with_filtering.txt")
        RetrievedPassageIDList.dump_trec_csv(
            retrieval_results=gathered_mined_negatives,
            fpath=fretrieved,
            system=args.cross_encoder,
        )
    logging.info("Done")


def load_negatives(
    negatives_path: str, dataset: RetrievalDataset, split: Split, prograss_bar: bool
) -> None:
    """Load the negatives and merge them into the corresponding labeled queries."""
    logging.info("Loading negatives")
    labeled_queries = dataset.get_labeled_queries(split)
    qid2query = LabeledQuery.build_qid2query(labeled_queries)
    qid2pids: Dict[str, List[str]] = {}
    with open(negatives_path) as f:
        for line in f:
            qid, _, pid, rank, score, system = line.strip().split()
            assert (
                qid in qid2query
            ), f"The mining results contain query ID not belonging to the {split} split"
            qid2pids.setdefault(qid, [])
            qid2pids[qid].append(pid)
    pids = {pid for passage_ids in qid2pids.values() for pid in passage_ids}
    pid2psg = {}
    for psg in tqdm.tqdm(
        dataset.collection_iter,
        desc="Locating negatives in collection",
        total=dataset.collection_size,
        disable=not prograss_bar,
    ):
        if psg.passage_id not in pids:
            continue
        pid2psg[psg.passage_id] = psg
    loaded_lqs = []
    for qid, pids in qid2pids.items():
        query = qid2query[qid]
        negatives = [
            JudgedPassage(query=query, passage=pid2psg[pid], judgement=0)
            for pid in pids
        ]
        lq = LabeledQuery(query=query, positives=[], negatives=negatives)
        loaded_lqs.append(lq)
    lqs_merged = LabeledQuery.merge(labeled_queries + loaded_lqs)
    dataset.set_labeled_queries(split=split, labeled_queries=lqs_merged)


if __name__ == "__main__":
    main()
