"""Build BM25 full-text search index on Milvus from Feverous wiki pages.

Supports resume via JSON checkpoint so interrupted runs can continue
without re-processing already-indexed documents.

Usage:
    # Fresh build (overwrite existing collection & checkpoint)
    python -m scripts.graph_check.build_index --upsert-workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from pymilvus import MilvusClient, MilvusException
from llama_index.core import Document
from llama_index.core.utils import iter_batch
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from tqdm import tqdm

from src.modules.datasets.feverous.database.feverous_db import FeverousDB
from src.modules.datasets.feverous.utils.feveous_utils import wiki_to_plain_text
from src.modules.datasets.feverous.utils.wiki_page import (
    WikiPage,
    WikiSection,
    WikiTable,
)

DEFAULT_CHECKPOINT = "build_index_checkpoint.json"


# ── Config ────────────────────────────────────────────────────────────────


@dataclass
class Config:
    document_path: str = "./datas/feverous_wikiv1.db"
    uri: str = "http://localhost:19531" # "https://milvus.vm.trungtd.work:19530"
    token: str = "root:Milvus"
    collection_name: str = "feverous_bm25"
    checkpoint_path: str = DEFAULT_CHECKPOINT
    batch_size: int = 2_000
    doc_batch_size: int = 100_000
    workers: int = mp.cpu_count()
    upsert_workers: int = 8
    fresh: bool = False


# ── Checkpoint ────────────────────────────────────────────────────────────


def load_checkpoint(path: str) -> set[str]:
    """Load already-processed doc_ids from checkpoint file."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data.get("processed_doc_ids", []))


def save_checkpoint(path: str, processed_ids: set[str]) -> None:
    """Atomically persist processed doc_ids."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"processed_doc_ids": sorted(processed_ids)}, f)
    os.replace(tmp, path)


# ── Document parsing ─────────────────────────────────────────────────────

def hash_string(s: str) -> str:
    """Generate a SHA-256 hash of the input string."""
    return hashlib.sha256(s.encode()).hexdigest()

def parse_wiki_page(doc_id: str, db: FeverousDB) -> list[Document]:
    """Parse a single wiki page into a list of section-level Documents."""
    page_json = db.get_doc_json(doc_id)
    wiki_page = WikiPage(doc_id, page_json)

    elements = wiki_page.get_page()
    title = wiki_to_plain_text(str(wiki_page.title))


    sections = []
    current_section = f"{title}\n"
    for element in elements:
        if isinstance(element, WikiTable): # skip table
            continue
        elif isinstance(element, WikiSection):
            sections.append(current_section)
            current_section = f"{title}\n"
            current_section += f"# {wiki_to_plain_text(str(element))}\n"
        else:
            current_section += f"{wiki_to_plain_text(str(element))} "
    sections.append(current_section) # add the last section

    return [
        Document(
            text=section,
            doc_id=f"{hash_string(doc_id)}_{i}_{hash_string(section)}",
        )
        for i, section in enumerate(sections)
    ]


# ── Multiprocessing worker ───────────────────────────────────────────────
# Pool initializer creates ONE db connection per worker process (not per
# task), which avoids the overhead of reconnecting for every doc_id.

_worker_db: FeverousDB | None = None


def _pool_initializer(db_path: str) -> None:
    global _worker_db
    _worker_db = FeverousDB(db_path)


def _worker_parse(doc_id: str) -> list[Document]:
    assert _worker_db is not None
    return parse_wiki_page(doc_id, _worker_db)


# ── Retry helper ─────────────────────────────────────────────────────────


def retry_on_milvus_error(
    func,
    *args,
    max_retries: int = 5,
    base_sleep: float = 2.0,
    backoff: float = 1.5,
    **kwargs,
):
    """Call *func* with exponential backoff on MilvusException."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except MilvusException as exc:
            if attempt == max_retries:
                raise
            sleep = base_sleep * (backoff ** (attempt - 1))
            tqdm.write(
                f"  ⚠ MilvusException ({exc}), "
                f"retry {attempt}/{max_retries} in {sleep:.1f}s …"
            )
            time.sleep(sleep)


# ── Upsert worker (ProcessPool) ──────────────────────────────────────────
# Each process gets its own MilvusVectorStore connection.

_upsert_store: MilvusVectorStore | None = None


def _upsert_pool_initializer(
    uri: str, collection_name: str, token: str, overwrite: bool
) -> None:
    global _upsert_store
    _upsert_store = MilvusVectorStore(
        uri=uri,
        collection_name=collection_name,
        token=token,
        enable_dense=False,
        enable_sparse=True,
        sparse_embedding_function=BM25BuiltInFunction(),
        overwrite=False,  # never overwrite from worker — only main process does that
        upsert_mode=not overwrite,
        use_async_client=False,
    )


def _upsert_worker(docs: list[Document]) -> int:
    """Insert a batch of documents. Returns count inserted."""
    assert _upsert_store is not None
    retry_on_milvus_error(_upsert_store.add, docs)
    return len(docs)


# ── Vector store factory u2500


def create_vector_store(cfg: Config, *, overwrite: bool) -> MilvusVectorStore:
    return MilvusVectorStore(
        uri=cfg.uri,
        collection_name=cfg.collection_name,
        token=cfg.token,
        enable_dense=False,
        enable_sparse=True,
        sparse_embedding_function=BM25BuiltInFunction(),
        overwrite=overwrite,
        upsert_mode=not overwrite,  # fresh build → insert (faster), resume → upsert (safe)
        use_async_client=False,
    )


# ── Main pipeline ────────────────────────────────────────────────────────


def build_index(cfg: Config) -> None:
    # 1. Resume logic
    processed_ids = set() if cfg.fresh else load_checkpoint(cfg.checkpoint_path)
    if processed_ids:
        tqdm.write(
            f"✓ Resuming — {len(processed_ids):,} doc_ids already done "
            f"({cfg.checkpoint_path})"
        )
    else:
        tqdm.write("● Starting fresh build …")

    # 2. Prepare doc_ids
    db = FeverousDB(cfg.document_path)
    all_doc_ids = list(db.get_doc_ids())
    remaining = [d for d in all_doc_ids if d not in processed_ids]
    total, todo = len(all_doc_ids), len(remaining)

    tqdm.write(f"  Total: {total:,} | Remaining: {todo:,} | Workers: {cfg.workers}")
    if not remaining:
        tqdm.write("✓ Nothing to do — all doc_ids already indexed.")
        return

    # 3. Vector store (overwrite only on fresh start)
    vector_store = create_vector_store(cfg, overwrite=cfg.fresh)

    # 4. Optimal chunksize for multiprocessing
    chunksize = max(1, math.ceil(cfg.doc_batch_size / (cfg.workers * 4)))

    # Split remaining into outer batches for checkpoint granularity
    outer_batches = [
        list(batch) for batch in iter_batch(remaining, cfg.doc_batch_size)
    ]
    total_sections = 0

    # ── Outer loop: one iteration = one checkpoint ──
    with tqdm(total=todo, desc="Total", unit="doc", position=0) as pbar_outer:
        for batch_list in outer_batches:
            # ── Inner loop A: Parse documents in parallel ──
            documents: list[Document] = []
            with mp.Pool(
                processes=cfg.workers,
                initializer=_pool_initializer,
                initargs=(cfg.document_path,),
            ) as pool:
                with tqdm(
                    total=len(batch_list),
                    desc="  Parse",
                    unit="doc",
                    leave=False,
                    position=1,
                ) as pbar_parse:
                    for result in pool.imap_unordered(
                        _worker_parse, batch_list, chunksize=chunksize
                    ):
                        documents.extend(result)
                        pbar_parse.update(1)
                        pbar_parse.set_postfix(sections=len(documents))

            # ── Inner loop B: Upsert into Milvus (concurrent processes) ──
            upsert_batches = [list(batch) for batch in iter_batch(documents, cfg.batch_size)]
            with tqdm(
                total=len(upsert_batches),
                desc=f"  Upsert total {len(documents):,} sections",
                unit="batch",
                leave=False,
                position=1,
            ) as pbar_upsert:
                with ProcessPoolExecutor(
                    max_workers=cfg.upsert_workers,
                    initializer=_upsert_pool_initializer,
                    initargs=(cfg.uri, cfg.collection_name, cfg.token, cfg.fresh),
                ) as executor:
                    futures = {
                        executor.submit(_upsert_worker, batch): i
                        for i, batch in enumerate(upsert_batches)
                    }
                    for future in as_completed(futures):
                        future.result()  # raise if failed
                        pbar_upsert.update(1)

            # ── Checkpoint ──
            processed_ids.update(batch_list)
            save_checkpoint(cfg.checkpoint_path, processed_ids)
            total_sections += len(documents)

            pbar_outer.update(len(batch_list))
            pbar_outer.set_postfix(
                sections=f"{total_sections:,}",
                done=f"{len(processed_ids):,}/{total:,}",
            )

    tqdm.write(
        f"\n✓ Done! {total_sections:,} sections upserted from "
        f"{len(processed_ids):,} doc_ids."
    )
    tqdm.write(f"  Checkpoint: {cfg.checkpoint_path}")

    # ── Flush & Stats ──
    tqdm.write("\n● Flushing collection …")
    client = MilvusClient(uri=cfg.uri, token=cfg.token)
    client.flush(cfg.collection_name)
    stats = client.get_collection_stats(cfg.collection_name)
    row_count = stats.get("row_count", "?")
    tqdm.write(f"  Collection: {cfg.collection_name}")
    tqdm.write(f"  Total rows in Milvus: {row_count:,}" if isinstance(row_count, int) else f"  Total rows in Milvus: {row_count}")
    tqdm.write(f"  Sections inserted this session: {total_sections:,}")
    tqdm.write(f"  Doc IDs processed (all time): {len(processed_ids):,}")
    client.close()


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Build BM25 index on Milvus")
    p.add_argument(
        "--fresh", action="store_true", help="Discard checkpoint & overwrite collection"
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--batch-size", type=int, default=10_000, help="Milvus upsert batch size"
    )
    p.add_argument(
        "--doc-batch-size",
        type=int,
        default=10_000,
        help="Doc IDs per outer checkpoint batch",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help=f"Parallel workers (default: {mp.cpu_count()} = all CPUs)",
    )
    p.add_argument(
        "--upsert-workers",
        type=int,
        default=4,
        help="Concurrent threads for Milvus upsert (default: 4)",
    )
    args = p.parse_args()
    return Config(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        doc_batch_size=args.doc_batch_size,
        workers=args.workers,
        upsert_workers=args.upsert_workers,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    build_index(parse_args())
