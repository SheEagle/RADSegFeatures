import argparse
import json
from pathlib import Path

from elasticsearch import Elasticsearch, helpers


def infer_dimensions(jsonl_path: Path) -> int:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for cluster in record.get("clusters", []):
                vector = cluster["v"] if isinstance(cluster, dict) else cluster
                if vector:
                    return len(vector)
    raise ValueError(f"Could not infer vector dimensions from {jsonl_path}")


def create_index(es: Elasticsearch, index_name: str, dim: int, recreate: bool) -> None:
    mapping = {
        "mappings": {
            "properties": {
                "image_id": {"type": "keyword"},
                "cluster_id": {"type": "integer"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }

    if es.indices.exists(index=index_name):
        if not recreate:
            print(f"Index '{index_name}' already exists, keeping it.")
            return
        print(f"Deleting existing index '{index_name}'...")
        es.indices.delete(index=index_name)

    print(f"Creating index '{index_name}' with dim={dim}...")
    es.indices.create(index=index_name, body=mapping)


def yield_docs(index_name: str, jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = record["image_id"]
            for fallback_idx, cluster in enumerate(record.get("clusters", [])):
                if isinstance(cluster, dict):
                    cluster_id = int(cluster.get("cluster_id", fallback_idx))
                    vector = cluster["v"]
                else:
                    cluster_id = fallback_idx
                    vector = cluster

                yield {
                    "_index": index_name,
                    "_id": f"{image_id}_{cluster_id}",
                    "_source": {
                        "image_id": image_id,
                        "cluster_id": cluster_id,
                        "vector": vector,
                    },
                }


def count_vectors(jsonl_path: Path) -> int:
    total = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            total += len(record.get("clusters", []))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk insert clustered feature JSONL into Elasticsearch.")
    parser.add_argument("--jsonl", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--index", type=str, required=True, help="Elasticsearch index name")
    parser.add_argument("--es_host", type=str, default="http://localhost:9200", help="Elasticsearch host URL")
    parser.add_argument("--dim", type=int, default=None, help="Vector dimension. Auto-inferred if omitted.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Bulk API chunk size")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the existing index before creating it again",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {jsonl_path}")

    es = Elasticsearch(args.es_host)
    if not es.ping():
        raise SystemExit(f"Could not connect to Elasticsearch at {args.es_host}")

    dim = args.dim or infer_dimensions(jsonl_path)
    create_index(es=es, index_name=args.index, dim=dim, recreate=args.recreate)

    total_vectors = count_vectors(jsonl_path)
    print(f"Streaming {total_vectors} vectors from {jsonl_path} to index '{args.index}'...")
    success, failed = helpers.bulk(
        es,
        yield_docs(args.index, jsonl_path),
        chunk_size=args.chunk_size,
        raise_on_error=False,
        stats_only=True,
    )
    print(f"Done. indexed={success} failed={failed}")


if __name__ == "__main__":
    main()
