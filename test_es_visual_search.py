import argparse
import math
import os
import zlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch
from PIL import Image
from redis import Redis
from vl_backends import create_backend


class TextSearchVisualizer:
    def __init__(
        self,
        es_host,
        es_index,
        redis_url,
        redis_key_prefix,
        image_root,
        backend_name="talk2dino",
        device="cpu",
        vector_field="vector",
        image_id_field="image_id",
        cluster_id_field="cluster_id",
        model_version="c-radio_v4-h",
        lang_model="siglip2-g",
        model_id=None,
    ):
        self.es = Elasticsearch(es_host)
        self.redis = Redis.from_url(redis_url)
        self.image_root = image_root
        self.redis_key_prefix = redis_key_prefix
        self.es_index = es_index
        self.vector_field = vector_field
        self.image_id_field = image_id_field
        self.cluster_id_field = cluster_id_field
        self.device = device
        self.backend_name = backend_name

        self.backend = create_backend(
            backend_name=backend_name,
            device=device,
            model_version=model_version,
            lang_model=lang_model,
            model_id=model_id,
        )
    @torch.no_grad()
    def encode_prompts(self, prompts):
        return self.backend.encode_text(prompts)

    @staticmethod
    def normalize_negative_prompts(negative_text):
        prompts = []
        if negative_text:
            if isinstance(negative_text, str):
                prompts = [part.strip() for part in negative_text.split(",") if part.strip()]
            elif isinstance(negative_text, list):
                prompts = [part.strip() for part in negative_text if str(part).strip()]

        if not prompts:
            prompts = ["background"]
        return prompts

    def knn_search_candidates(self, query_vector, candidate_k):
        response = self.es.search(
            index=self.es_index,
            knn={
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": candidate_k,
                "num_candidates": max(candidate_k * 4, 100),
            },
            _source=[self.image_id_field, self.cluster_id_field],
            size=candidate_k,
        )
        return response["hits"]["hits"]

    def direct_search_with_negatives(self, query_text, negative_text, candidate_k, top_k, temperature):
        negative_prompts = self.normalize_negative_prompts(negative_text)
        prompts = [query_text] + negative_prompts
        text_vectors = self.encode_prompts(prompts)

        positive_vector = text_vectors[0].detach().cpu().numpy().tolist()
        negative_vectors = [vec.detach().cpu().numpy().tolist() for vec in text_vectors[1:]]

        # Use the positive vector for ANN preselection, then let ES script_score apply the
        # same positive-vs-negative softmax logic used in search_demo directly inside search.
        preselected_hits = self.knn_search_candidates(query_vector=positive_vector, candidate_k=candidate_k)
        candidate_ids = [hit["_id"] for hit in preselected_hits]
        if not candidate_ids:
            return [], negative_prompts

        vector_field = self.vector_field
        script_source = f"""
double pos = cosineSimilarity(params.positive_vector, '{vector_field}');
double numer = Math.exp(pos * params.temperature);
double denom = numer;
for (neg in params.negative_vectors) {{
  double negScore = cosineSimilarity(neg, '{vector_field}');
  denom += Math.exp(negScore * params.temperature);
}}
return numer / denom;
"""

        response = self.es.search(
            index=self.es_index,
            query={
                "script_score": {
                    "query": {
                        "ids": {
                            "values": candidate_ids
                        }
                    },
                    "script": {
                        "source": script_source,
                        "params": {
                            "positive_vector": positive_vector,
                            "negative_vectors": negative_vectors,
                            "temperature": float(temperature),
                        },
                    },
                }
            },
            _source=[self.image_id_field, self.cluster_id_field],
            size=candidate_k,
        )

        scored_hits = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source", {})

            cluster_id = source.get(self.cluster_id_field)
            if cluster_id is None:
                hit_id = str(hit.get("_id", ""))
                if "_" in hit_id:
                    cluster_id = hit_id.rsplit("_", 1)[-1]
                else:
                    cluster_id = 0

            scored_hits.append(
                {
                    "image_id": source[self.image_id_field],
                    "cluster_id": int(cluster_id),
                    "score": float(hit.get("_score", 0.0)),
                    "raw_score": float(hit.get("_score", 0.0)),
                }
            )

        best_per_image = {}
        for item in scored_hits:
            image_id = item["image_id"]
            if image_id not in best_per_image or item["score"] > best_per_image[image_id]["score"]:
                best_per_image[image_id] = item

        results = sorted(best_per_image.values(), key=lambda item: item["score"], reverse=True)
        return results[:top_k], negative_prompts

    def load_feature_map(self, image_id):
        key = f"{self.redis_key_prefix}:{image_id}"
        payload = self.redis.hgetall(key)
        if not payload:
            raise KeyError(f"Redis key not found: {key}")

        height = int(payload[b"height"])
        width = int(payload[b"width"])
        dtype_name = payload[b"dtype"].decode("utf-8")
        encoding = payload[b"encoding"].decode("utf-8")
        data = payload[b"data"]

        if encoding != "zlib":
            raise ValueError(f"Unsupported encoding '{encoding}' for {key}")

        decoded = zlib.decompress(data)
        array = np.frombuffer(decoded, dtype=np.dtype(dtype_name)).reshape(height, width)
        return array

    def make_overlay(self, image, cluster_id_map, cluster_id):
        mask = (cluster_id_map == cluster_id).astype(np.uint8)
        if mask.sum() == 0:
            return np.array(image), None

        image_np = np.array(image)
        mask_resized = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

        overlay = image_np.copy().astype(np.float32)
        alpha = 0.45
        heat = cv2.GaussianBlur(mask_resized.astype(np.float32), (0, 0), sigmaX=8, sigmaY=8)
        if heat.max() > 0:
            heat = heat / heat.max()
        heat_uint8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        overlay[mask_resized.astype(bool)] = (
            overlay[mask_resized.astype(bool)] * (1.0 - alpha)
            + heatmap_rgb[mask_resized.astype(bool)] * alpha
        )
        overlay = overlay.astype(np.uint8)
        return overlay, mask_resized

    def resolve_image_path(self, image_id):
        image_path = os.path.join(self.image_root, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path

    def visualize_results(self, results, query_text, negative_prompts, output_path=None):
        if not results:
            print("No matches found.")
            return

        cols = min(3, len(results))
        rows = math.ceil(len(results) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
        axes = np.atleast_1d(axes).reshape(rows, cols)

        for ax in axes.flat:
            ax.axis("off")

        for idx, result in enumerate(results):
            ax = axes[idx // cols, idx % cols]
            image_path = self.resolve_image_path(result["image_id"])
            image = Image.open(image_path).convert("RGB")
            cluster_id_map = self.load_feature_map(result["image_id"])
            overlay, _ = self.make_overlay(image, cluster_id_map, result["cluster_id"])

            ax.imshow(overlay)

            neg_text = ", ".join(negative_prompts)
            ax.set_title(
                f"{result['image_id']}\n"
                f"cluster={result['cluster_id']} score={result['score']:.4f}\n"
                f"query='{query_text}' vs [{neg_text}]",
                fontsize=11,
            )
            ax.axis("off")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()


def infer_vector_field(index_name):
    if index_name == "radseg_features":
        return "cluster_vector"
    return "vector"


def main():
    parser = argparse.ArgumentParser(description="Test Elasticsearch text search with Redis-backed feature-map visualization.")
    parser.add_argument("query", type=str, default="river",help="Positive text query")
    parser.add_argument("--negative_text", type=str, default="background", help="Comma-separated negative prompts")
    parser.add_argument("--top_k", type=int, default=6, help="Number of unique images to visualize")
    parser.add_argument("--candidate_k", type=int, default=120, help="Number of candidate clusters retrieved from ES before direct negative scoring")
    parser.add_argument("--temperature", type=float, default=20.0, help="Softmax temperature for reranking")
    parser.add_argument("--es_host", type=str, default="http://localhost:9201", help="Elasticsearch host")
    parser.add_argument("--es_index", type=str, default="radseg_images", help="Elasticsearch index name")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0", help="Redis connection URL")
    parser.add_argument("--redis_key_prefix", type=str, default="fm", help="Redis key prefix used for feature maps")
    parser.add_argument("--image_root", type=str, default="images", help="Directory containing original images")
    parser.add_argument(
        "--backend",
        type=str,
        default="talk2dino",
        choices=["talk2dino", "radseg"],
        help="Vision-language backend used to encode the query text",
    )
    parser.add_argument("--model_version", type=str, default="c-radio_v4-h", help="RADSeg model version")
    parser.add_argument("--lang_model", type=str, default="siglip2-g", help="RADSeg language model")
    parser.add_argument("--model_id", type=str, default="lorebianchi98/Talk2DINO-ViTL", help="Talk2DINO model id")
    parser.add_argument("--device", type=str, default="cuda", help="Device for text encoding")
    parser.add_argument("--vector_field", type=str, default=None, help="Override ES vector field name")
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to save the matplotlib figure")
    args = parser.parse_args()

    vector_field = args.vector_field or infer_vector_field(args.es_index)

    visualizer = TextSearchVisualizer(
        es_host=args.es_host,
        es_index=args.es_index,
        redis_url=args.redis_url,
        redis_key_prefix=args.redis_key_prefix,
        image_root=args.image_root,
        backend_name=args.backend,
        model_version=args.model_version,
        lang_model=args.lang_model,
        model_id=args.model_id,
        device=args.device,
        vector_field=vector_field,
    )

    if not visualizer.es.ping():
        raise SystemExit(f"Could not connect to Elasticsearch at {args.es_host}")
    visualizer.redis.ping()

    results, negative_prompts = visualizer.direct_search_with_negatives(
        query_text=args.query,
        negative_text=args.negative_text,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    print(f"Top {len(results)} results for '{args.query}':")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. image={result['image_id']} cluster={result['cluster_id']} "
            f"score={result['score']:.4f} raw_es={result['raw_score']:.4f}"
        )

    visualizer.visualize_results(
        results=results,
        query_text=args.query,
        negative_prompts=negative_prompts,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
