from __future__ import annotations
from dataclasses import dataclass
import logging
import os
from typing import Dict, Iterable, List, Optional, Set, Type
from clddp.reranker import Reranker, RerankerInputExample
from more_itertools import chunked
import torch
import tqdm
from clddp.dm import (
    LabeledQuery,
    Passage,
    Query,
    RetrievedPassageIDList,
    ScoredPassageID,
)
from clddp.args.search import SearchArguments
from clddp.dataloader import load_dataset
from clddp.utils import (
    NINF,
    get_rank,
    all_gather_object,
    is_device_zero,
    set_logger_format,
    split_data,
    split_data_size,
    parse_cli,
    initialize_ddp,
)
from clddp.retriever import Retriever


@dataclass
class SearchOutput:
    wid: int  # worker index
    topk_indices: torch.LongTensor  # (nqueries, topk)
    topk_values: torch.Tensor  # (nqueries, topk)

    def update(
        self,
        base: Optional[int],
        similarity_matrix: torch.Tensor,
        nqueries: int,
        topk: int,
    ) -> int:
        nentries: int = similarity_matrix.shape[1]  # either #passages or #documents
        assert similarity_matrix.shape[0] == nqueries
        self.topk_values, topk_setoffs = torch.cat(
            [self.topk_values, similarity_matrix], dim=-1
        ).topk(k=topk, dim=1, largest=True, sorted=False)
        new_indices = (
            torch.arange(nentries)
            .to(self.topk_indices)
            .unsqueeze(0)
            .expand(nqueries, nentries)
        ) + base
        self.topk_indices = torch.cat([self.topk_indices, new_indices], dim=-1).gather(
            dim=1, index=topk_setoffs
        )
        return nentries

    @classmethod
    def merge(
        cls: Type[SearchOutput], souts: List[SearchOutput], topk: int
    ) -> SearchOutput:
        assert len(souts)
        topk_values, topk_setoffs = torch.cat(
            [sout.topk_values for sout in souts], dim=-1
        ).topk(k=topk, dim=1, largest=True, sorted=False)
        topk_indices = torch.cat([sout.topk_indices for sout in souts], dim=-1).gather(
            dim=1, index=topk_setoffs
        )
        return SearchOutput(
            wid=-1,
            topk_indices=topk_indices,
            topk_values=topk_values,
        )

    def to_retrieved_passage_id_lists(
        self, queries: List[Query], passage_ids: List[str], pbar: bool
    ) -> List[RetrievedPassageIDList]:
        topk_indices = self.topk_indices.cpu().tolist()
        topk_values = self.topk_values.cpu().tolist()
        retrieval_results: List[RetrievedPassageIDList] = []
        for i, query in enumerate(
            tqdm.tqdm(queries, desc="Formatting results", disable=not pbar)
        ):
            spids: List[ScoredPassageID] = []
            pids = map(
                lambda topk_index: passage_ids[topk_index],
                topk_indices[i],
            )
            for pid, score in zip(pids, topk_values[i]):
                spids.append(ScoredPassageID(passage_id=pid, score=score))
            retrieval_results.append(
                RetrievedPassageIDList(
                    query_id=query.query_id, scored_passage_ids=spids
                )
            )
        return retrieval_results

    def cpu(self) -> SearchOutput:
        return SearchOutput(
            wid=self.wid,
            topk_indices=self.topk_indices.cpu(),
            topk_values=self.topk_values.cpu(),
        )


@torch.no_grad()
def search_single_device(
    retriever: Retriever,
    queries: List[Query],
    collection_iter: Iterable[Passage],
    collection_size: int,
    topk: int,
    batch_size: int,
    fp16: bool,
    wid: int = 0,
    pid2allowed_queries: Optional[Dict[str, Set[int]]] = None,
    show_pbar: bool = True,
) -> List[RetrievedPassageIDList]:
    retriever.eval()
    qembs = retriever.encode_queries(queries=queries, batch_size=batch_size)
    search_output = SearchOutput(
        wid=wid,
        topk_indices=torch.zeros(len(qembs), topk, dtype=torch.int64).to(
            retriever.device
        ),
        topk_values=torch.full((len(qembs), topk), NINF).to(retriever.device),
    )
    batches = chunked(iterable=collection_iter, n=batch_size)
    pids: List[str] = []
    pbar = tqdm.tqdm(total=collection_size, desc="Searching", disable=not show_pbar)
    for batch in batches:
        with torch.cuda.amp.autocast(enabled=fp16):
            pembs, mask = retriever.encode_passages(
                passages=batch, batch_size=batch_size
            )
        sim_mtrx = retriever.similarity_function(
            query_embeddings=qembs, passage_embeddings=pembs, passage_mask=mask
        )
        if pid2allowed_queries:  # For scoped search:
            for col, psg in enumerate(batch):
                allowed_queries = list(pid2allowed_queries.get(psg.passage_id, {}))
                allowed = sim_mtrx[allowed_queries, col]
                sim_mtrx[:, col] = NINF
                sim_mtrx[allowed_queries, col] = allowed
        nentires = search_output.update(
            base=len(pids),
            similarity_matrix=sim_mtrx,
            nqueries=len(qembs),
            topk=topk,
        )
        pids.extend(psg.passage_id for psg in batch)
        pbar.update(nentires)
    return search_output.to_retrieved_passage_id_lists(
        queries=queries, passage_ids=pids, pbar=show_pbar
    )


def search(
    retriever: Retriever,
    collection_iter: Iterable[Passage],
    collection_size: int,
    queries: List[Query],
    topk: int,
    per_device_eval_batch_size: int,
    fp16: bool,
    passage_scopes: Optional[
        List[Set[str]]
    ] = None,  # For each query, which pids are allowed
) -> List[RetrievedPassageIDList]:
    # Doing search:
    pid2allowed_queries: Optional[Dict[str, Set[int]]] = None
    if passage_scopes:
        pid2allowed_queries = {}
        for query_i, scope in enumerate(
            tqdm.tqdm(
                passage_scopes,
                desc="Processing passage scopes",
                disable=not is_device_zero(),
            )
        ):
            for pid in scope:
                pid2allowed_queries.setdefault(pid, set())
                pid2allowed_queries[pid].add(query_i)
    rank = get_rank()
    split_collection_iter = split_data(data=collection_iter)
    split_collection_size = split_data_size(collection_size)
    retrieved = search_single_device(
        retriever=retriever,
        queries=queries,
        collection_iter=split_collection_iter,
        collection_size=split_collection_size,
        topk=topk,
        batch_size=per_device_eval_batch_size,
        fp16=fp16,
        wid=rank,
        pid2allowed_queries=pid2allowed_queries,
        show_pbar=is_device_zero(),
    )
    all_retrieved = sum(all_gather_object(retrieved), [])
    qid2rpidls: Dict[str, List[RetrievedPassageIDList]] = {}
    for rpidl in all_retrieved:
        qid2rpidls.setdefault(rpidl.query_id, [])
        qid2rpidls[rpidl.query_id].append(rpidl)
    merged_rpidls = []
    for rpidls in tqdm.tqdm(
        list(qid2rpidls.values()),
        desc="Merging search results",
        disable=not is_device_zero(),
    ):
        rpidl = RetrievedPassageIDList.merge(rpidls=rpidls, topk=topk)
        merged_rpidls.append(rpidl)
    return merged_rpidls


def rerank(
    retrieval_results: List[RetrievedPassageIDList],
    reranker: Reranker,
    collection_iter: Iterable[Passage],
    collection_size: int,
    queries: List[Query],
    per_device_eval_batch_size: int,
    fp16: bool,
) -> List[RetrievedPassageIDList]:
    # Prepare the data:
    qid2query = {q.query_id: q for q in queries}
    pids = {
        spid.passage_id
        for rpidl in retrieval_results
        for spid in rpidl.scored_passage_ids
    }
    pid2psg = {}
    for psg in tqdm.tqdm(
        collection_iter,
        total=collection_size,
        desc="Locating passages",
        disable=not is_device_zero(),
    ):
        if psg.passage_id in pids:
            pid2psg[psg.passage_id] = psg
    split_retrieval_results_iter = split_data(data=retrieval_results)
    split_retrieval_results_size = split_data_size(len(retrieval_results))

    # Reranking:
    reranked: List[RetrievedPassageIDList] = []
    for rpidl in tqdm.tqdm(
        split_retrieval_results_iter,
        total=split_retrieval_results_size,
        desc="Reranking",
        disable=not is_device_zero(),
    ):
        query = qid2query[rpidl.query_id]
        candidates: List[Passage] = [
            pid2psg[spid.passage_id] for spid in rpidl.scored_passage_ids
        ]
        scores = []
        for b in range(0, len(candidates), per_device_eval_batch_size):
            e = b + per_device_eval_batch_size
            batch_candidates = candidates[b:e]
            batch = RerankerInputExample(query=query, passages=batch_candidates)
            with torch.cuda.amp.autocast(enabled=fp16):
                batch_scores = (
                    reranker.predict([batch]).squeeze().tolist()
                )  # (batch_size,)
            scores.extend(batch_scores)
        scored_passage_ids = [
            ScoredPassageID(passage_id=psg.passage_id, score=score)
            for psg, score in zip(candidates, scores)
        ]
        reranked_rpidl = RetrievedPassageIDList(
            query_id=query.query_id, scored_passage_ids=scored_passage_ids
        )
        reranked.append(reranked_rpidl)
    reranked = sum(all_gather_object(reranked), [])

    # Sort back to the original order wrt. the query IDs:
    qid2reranking_result = {rpidl.query_id: rpidl for rpidl in reranked}
    sorted_reranked = [
        qid2reranking_result[rpidl.query_id] for rpidl in retrieval_results
    ]
    return sorted_reranked


def main(args: Optional[SearchArguments] = None) -> List[RetrievedPassageIDList]:
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    if args is None:
        args = parse_cli(SearchArguments)
    if is_device_zero():
        args.dump_arguments()

    retriever = Retriever.from_pretrained(args.checkpoint_dir)
    if args.query_prompt is not None:
        retriever.set_query_prompt(args.query_prompt)
    if args.passage_prompt is not None:
        retriever.set_passage_prompt(args.passage_prompt)
    retriever.set_device(get_rank())
    dataset = load_dataset(
        enable=True, dataloader_name=args.dataloader, data_name_or_path=args.data_dir
    )
    labeled_queries = dataset.get_labeled_queries(args.split)
    queries = LabeledQuery.get_unique_queries(labeled_queries)
    ranking_results = search(
        retriever=retriever,
        collection_iter=dataset.collection_iter,
        collection_size=dataset.collection_size,
        queries=queries,
        topk=args.topk,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
    )
    system = "retriever"
    if args.reranker_checkpoint_dir:
        reranker = Reranker.from_pretrained(args.reranker_checkpoint_dir)
        ranking_results = rerank(
            retrieval_results=ranking_results,
            reranker=reranker,
            collection_iter=dataset.collection_iter,
            collection_size=dataset.collection_size,
            queries=queries,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.fp16,
        )
        system = "retriever+reranker"

    if is_device_zero():
        franked = os.path.join(args.output_dir, "ranking_results.txt")
        RetrievedPassageIDList.dump_trec_csv(
            retrieval_results=ranking_results, fpath=franked, system=system
        )
    logging.info("Done")
    return ranking_results


if __name__ == "__main__":
    main()
