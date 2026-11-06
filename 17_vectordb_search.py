"""Search Qdrant by semantic similarity with optional metadata filters."""
import argparse
from datetime import timedelta
from importlib import import_module

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ingest = import_module("17_vectordb_ingest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--url", default=ingest.DEFAULT_URL)
    parser.add_argument("--collection", default=ingest.DEFAULT_COLLECTION)
    parser.add_argument("--lang")
    parser.add_argument("--source")
    parser.add_argument("--date-from", type=ingest.date.fromisoformat)
    parser.add_argument("--date-to", type=ingest.date.fromisoformat)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1 or not args.query.strip():
        parser.error("Provide a nonempty query and a positive --top-k.")
    if args.date_from and args.date_to and args.date_from > args.date_to:
        parser.error("--date-from must be on or before --date-to.")

    conditions = [
        models.FieldCondition(key=field, match=models.MatchValue(value=value))
        for field, value in [("lang", args.lang), ("source", args.source)]
        if value is not None
    ]
    if args.date_from or args.date_to:
        conditions.append(models.FieldCondition(
            key="date",
            range=models.DatetimeRange(
                gte=f"{args.date_from}T00:00:00Z" if args.date_from else None,
                lt=f"{args.date_to + timedelta(days=1)}T00:00:00Z" if args.date_to else None,
            ),
        ))

    client = QdrantClient(url=args.url)
    try:
        if not client.collection_exists(args.collection):
            parser.error("Collection does not exist. Run 17_vectordb_ingest.py first.")
        encoder = SentenceTransformer(ingest.MODEL_NAME)
        vector = encoder.encode(args.query, normalize_embeddings=True).tolist()
        hits = client.query_points(
            args.collection,
            query=vector,
            query_filter=models.Filter(must=conditions) if conditions else None,
            limit=args.top_k,
            with_payload=True,
        ).points

    finally:
        client.close()
    
    if not hits:
        print("Filtrlarga mos natija topilmadi.")
        return

    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        date_label = (payload.get("date") or "")[:10] or "sana noma’lum"
        title = payload.get("title") or "Sarlavha noma’lum"
        source = payload.get("source") or "manba noma’lum"
        lang = payload.get("lang") or "til noma’lum"

        print(f"{rank}. {title} | cosine={hit.score:.3f}")
        print(f"   {source} | {date_label} | {lang}")
        if payload.get("article"):
            print(f"   Modda: {payload['article']}")
        if payload.get("source_url"):
            print(f"   Manba: {payload['source_url']}")
        print(f"   {payload.get('text', '')}")
        print()

if __name__ == "__main__":
    main()
