"""Build ChromaDB vector index from shabads.jsonl.

Usage:
    python -m src.embed [--reset]

Options:
    --reset   Drop and recreate the collection before upserting.
"""

import argparse
import json
import logging
import os
from typing import Any

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    SHABADS_FILE,
    WINDOW_OVERLAP,
    WINDOW_SIZE,
)
from src.corpus import ShabadLine, load_shabads, make_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _make_passage_text(window: list[ShabadLine]) -> str:
    """Create embed text from a window of lines: 'transliteration | translation_en' per line."""
    parts = []
    for line in window:
        parts.append(f"{line.transliteration} | {line.translation_en}")
    return "\n".join(parts)


def _make_passage_document(window: list[ShabadLine]) -> str:
    """Stored document: JSON with parallel arrays for gurmukhi and translation_en."""
    return json.dumps(
        {
            "gurmukhi": [line.gurmukhi for line in window],
            "translation_en": [line.translation_en for line in window],
        },
        ensure_ascii=False,
    )


def build_index(reset: bool = False) -> None:
    """Load shabads.jsonl, window, embed, and upsert into ChromaDB."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "chromadb and sentence-transformers are required. Run: pip install -r requirements.txt"
        ) from exc

    if not os.path.exists(SHABADS_FILE):
        raise FileNotFoundError(
            f"Corpus not found at {SHABADS_FILE}. Run `python -m src.ingest` first."
        )

    os.makedirs(CHROMA_DIR, exist_ok=True)

    logger.info("Loading embedding model %s …", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    logger.info("Connecting to ChromaDB at %s …", CHROMA_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            logger.info("Dropped existing collection '%s'", CHROMA_COLLECTION)
        except Exception:  # noqa: BLE001
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Collect all passages
    ids: list[str] = []
    texts: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    logger.info("Windowing shabads from %s …", SHABADS_FILE)
    for shabad in load_shabads(SHABADS_FILE):
        windows = make_windows(shabad.lines, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP)
        for win_idx, window in enumerate(windows):
            passage_id = f"{shabad.shabad_id}-{win_idx}"
            ids.append(passage_id)
            texts.append(_make_passage_text(window))
            documents.append(_make_passage_document(window))
            metadatas.append(
                {
                    "shabad_id": shabad.shabad_id,
                    "ang": shabad.ang,
                    "raag": shabad.raag,
                    "writer": shabad.writer,
                    "window": win_idx,
                }
            )

    total_passages = len(ids)
    logger.info("Total passages to embed: %d", total_passages)

    # Embed and upsert in batches
    for batch_start in range(0, total_passages, EMBED_BATCH_SIZE):
        batch_end = min(batch_start + EMBED_BATCH_SIZE, total_passages)
        batch_ids = ids[batch_start:batch_end]
        batch_texts = texts[batch_start:batch_end]
        batch_docs = documents[batch_start:batch_end]
        batch_metas = metadatas[batch_start:batch_end]

        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()

        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_metas,
        )

        if (batch_end % 1000 == 0) or batch_end == total_passages:
            logger.info("Upserted %d / %d passages", batch_end, total_passages)

    final_count = collection.count()
    logger.info("Index complete. Collection '%s' has %d passages.", CHROMA_COLLECTION, final_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from shabads.jsonl")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection",
    )
    args = parser.parse_args()
    build_index(reset=args.reset)


if __name__ == "__main__":
    main()
