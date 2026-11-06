"""Load documents or pre-chunked records into a local Qdrant collection."""
import argparse
import json
from datetime import date
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "rag17_multilingual_minilm"


def iso_date(value):
    return date.fromisoformat(value).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("documents.json"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--source", help="Default source for rows without source")
    parser.add_argument("--date", type=iso_date, help="Default ISO date for rows without date")
    parser.add_argument("--lang", default="uz", help="Default language for rows without lang")
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8-sig"))
    payloads = {}
    for row in records:
        source = row.get("source", args.source or args.input.name)
        document_id = row["id"]
        chunk_id = row.get("chunk_id", 0)
        key = json.dumps([source, str(document_id), str(chunk_id)], ensure_ascii=False)
        point_id = str(uuid5(NAMESPACE_URL, key))
        if point_id in payloads:
            parser.error(f"Duplicate source/document/chunk: {key}")
        raw_date = row.get("date") or args.date
        payloads[point_id] = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "title": row.get("title", ""),
            "text": row["text"],
            "source": source,
            "date": iso_date(raw_date) + "T00:00:00Z" if raw_date else None,
            "lang": row.get("lang") or args.lang,
            "article": row.get("article"),
            "source_url": row.get("source_url"),
        }
    if not payloads:
        parser.error("Input contains no records.")

    encoder = SentenceTransformer(MODEL_NAME)
    vectors = encoder.encode(
        [f"{item['title']}\n{item['text']}" for item in payloads.values()],
        normalize_embeddings=True,
    )
    client = QdrantClient(url=args.url)
    try:
        if not client.collection_exists(args.collection):
            client.create_collection(
                args.collection,
                vectors_config=models.VectorParams(
                    size=vectors.shape[1], distance=models.Distance.COSINE
                ),
            )
        points = [
            models.PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            for (point_id, payload), vector in zip(payloads.items(), vectors, strict=True)
        ]
        for start in range(0, len(points), 128):
            client.upsert(args.collection, points=points[start:start + 128], wait=True)
        total = client.count(args.collection, exact=True).count

    finally:
            client.close()
    print(f"Upserted: {len(points)}. Collection total: {total}.")

    


if __name__ == "__main__":
    main()
