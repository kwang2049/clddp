import logging
import os
import pickle
from typing import Iterable, Iterator, List, Type, TypeVar
import git
import numpy as np
import torch.distributed as dist
import torch
import tqdm
from transformers import HfArgumentParser


NINF = -1e4


def is_device_zero() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


def set_logger_format() -> None:
    logging.basicConfig(level=logging.INFO)
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)


OBJ_TYPE = TypeVar("OBJ_TYPE")


def broadcast_object(obj: OBJ_TYPE) -> OBJ_TYPE:
    """Broadcast a Python object across all the nodes."""
    # Collect object from replicas and form a list
    if not dist.is_initialized():
        return obj

    rank = dist.get_rank()
    obj_size: torch.LongTensor = torch.zeros(1).long().to(rank)  # long not int!
    if rank == 0:
        data = pickle.dumps(obj)
        data_length = len(data)
        data = data_length.to_bytes(4, "big") + data
        data = np.frombuffer(data, dtype=np.uint8)
        obj_size += len(data)
        tensorized: torch.Tensor = torch.from_numpy(data).to(rank)
        logging.info(f"Going to broacast {obj_size.item() / 2**20:.1f}MB")
    dist.broadcast(obj_size, 0)
    if rank != 0:
        tensorized = torch.zeros(obj_size.item(), dtype=torch.uint8).to(rank)
    dist.broadcast(tensorized, 0)
    tensorized_numpy: np.ndarray = tensorized.cpu().numpy()
    data = tensorized_numpy.tobytes()
    del tensorized
    torch.cuda.empty_cache()
    length = int.from_bytes(data[:4], "big")
    data = data[4 : length + 4]
    obj: OBJ_TYPE = pickle.loads(data)
    return obj


def all_gather_object(obj: OBJ_TYPE) -> List[OBJ_TYPE]:
    """Broadcast a Python object across all the nodes."""
    if not dist.is_initialized():
        return [obj]

    # Collect object from replicas and form a list
    ngpus = dist.get_world_size()
    obj_list = [None for _ in range(ngpus)]
    dist.all_gather_object(obj_list, obj)
    return obj_list


def split_data(data: Iterable[OBJ_TYPE]) -> Iterable[OBJ_TYPE]:
    if not dist.is_initialized():
        for datum in data:
            yield datum

    rank = dist.get_rank()
    ngpus = dist.get_world_size()
    for i, datum in enumerate(data):
        if i % ngpus == rank:
            yield datum


def split_data_size(data_size: int) -> int:
    if not dist.is_initialized():
        return data_size

    rank = dist.get_rank()
    ngpus = dist.get_world_size()
    return data_size // ngpus + int(rank < data_size % ngpus)


def dist_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized()
    ngpus = dist.get_world_size()
    rank = dist.get_rank()

    if t is None:
        return None
    t = t.contiguous()

    all_tensors = [torch.empty_like(t) for _ in range(ngpus)]
    dist.all_gather(all_tensors, t)

    all_tensors[rank] = t
    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors


def tqdm_ropen(fpath: str, desc: str, pbar: bool) -> Iterator[str]:
    """tqdm + open with r mode."""
    if desc is None:
        desc = f"Loading from {fpath}"

    with open(fpath, "r") as f:
        nlines = sum(1 for _ in f)

    with tqdm.tqdm(open(fpath, "r"), desc=desc, total=nlines, disable=not pbar) as f:
        for line in f:
            yield line


def set_logger_format(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level)
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_ngpus() -> int:
    if not dist.is_initialized():
        return int(torch.cuda.is_available())
    return dist.get_world_size()


def initialize_ddp(ddp_timeout: int = 360000) -> None:
    from transformers import TrainingArguments

    TrainingArguments(
        "dummy", ddp_timeout=ddp_timeout
    )  # This will invoke dist.init_process_group
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_env.split(",")[get_rank()]
    assert dist.is_initialized()


def get_commit_hash() -> str:
    """Return the HEAD commit hash."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def parse_cli(arguments_class: Type[OBJ_TYPE]) -> OBJ_TYPE:
    parser = HfArgumentParser(arguments_class)
    args = arguments_class(**vars(parser.parse_args()))
    return args


def colbert_score_reduce(
    scores_padded: torch.Tensor,
    D_mask: torch.Tensor,
    interaction: str = "colbert",
    query_maxlen: int = 64,
):
    """Copied and modified from ColBERT."""
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores: torch.Tensor = scores_padded.max(1).values
    assert interaction in ["colbert", "flipr"]
    if interaction == "flipr":
        K1 = query_maxlen // 2
        K2 = 8
        A = scores[:, :query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0
        if K2 <= scores.size(1) - query_maxlen:
            B = scores[:, query_maxlen:].topk(K2, dim=-1).values.sum(1)
        return A + B
    return scores.sum(-1)


def colbert_score(Q: torch.Tensor, D_padded: torch.Tensor, D_mask: torch.BoolTensor):
    """
    Copied and modified from ColBERT.
    Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
    If Q.size(0) is 1, the matrix will be compared with all passages.
    Otherwise, each query matrix will be compared against the *aligned* passage.

    EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]
    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
    return colbert_score_reduce(scores, D_mask)
