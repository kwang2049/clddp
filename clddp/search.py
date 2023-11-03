from __future__ import annotations
from dataclasses import dataclass
import logging
import os
from typing import Dict, Iterable, List, Optional, Type
from more_itertools import chunked
import torch
import tqdm
from clddp.dm import (
    JudgedPassage,
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
            pembs = retriever.encode_passages(passages=batch, batch_size=batch_size)
        sim_mtrx = retriever.similarity_function(
            query_embeddings=qembs, passage_embeddings=pembs
        )
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
) -> List[RetrievedPassageIDList]:
    # Doing search:
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


if __name__ == "__main__":
    from clddp.utils import parse_cli, initialize_ddp

    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
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
    jpsgs = dataset.get_judged_passages(args.split)
    queries = JudgedPassage.get_unique_queries(jpsgs)
    retrieval_results = search(
        retriever=retriever,
        collection_iter=dataset.collection_iter,
        collection_size=dataset.collection_size,
        queries=queries,
        topk=args.topk,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
    )
    if is_device_zero():
        fretrieved = os.path.join(args.output_dir, "retrieval_results.txt")
        RetrievedPassageIDList.dump_trec_csv(
            retrieval_results=retrieval_results, fpath=fretrieved
        )
    logging.info("Done")
