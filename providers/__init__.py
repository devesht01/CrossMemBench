from __future__ import annotations

from typing import Any

from .base import BaseMemoryProvider
from .dense_retrieval import DenseRetrievalMemoryProvider
from .full_dump import FullDumpMemoryProvider
from .no_mem import NoMemProvider


def _create_no_mem(memory_cfg: dict[str, Any]) -> NoMemProvider:
    return NoMemProvider(top_k=memory_cfg["top_k"])


def _create_full_dump(memory_cfg: dict[str, Any]) -> FullDumpMemoryProvider:
    extra = memory_cfg["full_dump"]
    return FullDumpMemoryProvider(
        top_k=memory_cfg["top_k"],
        insert_inferred=extra["insert_inferred"],
    )


def _create_dense_retrieval(memory_cfg: dict[str, Any]) -> DenseRetrievalMemoryProvider:
    extra = memory_cfg["dense_retrieval"]
    return DenseRetrievalMemoryProvider(
        top_k=memory_cfg["top_k"],
        embedding_model=extra["embedding_model"],
        batch_size=extra["batch_size"],
    )


MEMORY_PROVIDERS = {
    "no_mem": _create_no_mem,
    "full_dump": _create_full_dump,
    "dense_retrieval": _create_dense_retrieval,
}
