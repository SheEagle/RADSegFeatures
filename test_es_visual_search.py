import argparse
import csv
import math
import os
import re
import unicodedata
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


GENERAL_QUERY_TERMS = {
    "architecture",
    "building",
    "bridge",
    "canal",
    "castle",
    "cathedral",
    "church",
    "city",
    "harbor",
    "lake",
    "mountain",
    "park",
    "river",
    "road",
    "square",
    "station",
    "street",
    "tower",
    "town",
    "village",
    "water",
}

GENERAL_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "in",
    "near",
    "of",
    "on",
    "over",
    "the",
    "under",
    "with",
}

GENERAL_QUERY_MIN_VISUAL_TERMS = 2

METADATA_SEARCH_FIELDS = [
    "landmarks_identified",
    "final_place",
    "description",
    "final_city",
    "final_country",
    "transcription",
]


def slugify_for_filename(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "query"


def build_default_output_path(backend, es_index, query_text, result_mode):
    os.makedirs("scratch", exist_ok=True)
    safe_query = slugify_for_filename(query_text)
    suffix = "cluster_mode" if result_mode == "cluster" else "heatmap"
    filename = f"{backend}_{es_index}_{safe_query}_{suffix}.png"
    return os.path.join("scratch", filename)


def normalize_text(text):
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.lower()).strip()


def printable(text):
    return str(text).encode("ascii", errors="backslashreplace").decode("ascii")


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize_text(text))


class MetadataQueryRouter:
    """Route broad visual concepts through vector search and named places through metadata-filtered reranking."""

    def __init__(self, metadata_csv, max_images=500):
        self.metadata_csv = metadata_csv
        self.max_images = max_images
        self.rows = []
        if metadata_csv and os.path.exists(metadata_csv):
            with open(metadata_csv, "r", encoding="latin1", newline="") as handle:
                self.rows = list(csv.DictReader(handle))

    def route(self, query_text, mode="auto"):
        if mode == "off" or not self.rows:
            return {
                "query_type": "general",
                "candidate_image_ids": None,
                "reason": "metadata filtering disabled or metadata CSV missing",
                "support": 0,
                "examples": [],
            }

        scored = self._score_metadata_matches(query_text)
        candidates = [item for item in scored if item["score"] >= 3.0]
        query_tokens = set(tokenize(query_text))
        content_tokens = query_tokens - GENERAL_QUERY_STOPWORDS
        visual_token_count = len(content_tokens & GENERAL_QUERY_TERMS)
        is_generic = bool(content_tokens) and (
            content_tokens.issubset(GENERAL_QUERY_TERMS)
            or visual_token_count >= GENERAL_QUERY_MIN_VISUAL_TERMS
        )
        should_filter = mode == "force" or (bool(candidates) and not is_generic)

        if not should_filter:
            return {
                "query_type": "general",
                "candidate_image_ids": None,
                "reason": "broad visual concept; using pure vector search",
                "support": len(candidates),
                "examples": candidates[:5],
            }

        candidate_image_ids = [item["image_id"] for item in candidates[: self.max_images]]
        return {
            "query_type": "specific",
            "candidate_image_ids": candidate_image_ids,
            "reason": "named place/landmark matched metadata; filtering candidates before vector rerank",
            "support": len(candidates),
            "examples": candidates[:5],
        }

    def _score_metadata_matches(self, query_text):
        query_norm = normalize_text(query_text)
        query_tokens = set(tokenize(query_text))
        if not query_tokens:
            return []

        scored = []
        for row in self.rows:
            image_id = (row.get("image_filename") or "").strip()
            if not image_id:
                continue

            score = 0.0
            matched_fields = []
            for field in METADATA_SEARCH_FIELDS:
                value = row.get(field) or ""
                value_norm = normalize_text(value)
                if not value_norm:
                    continue

                field_tokens = set(tokenize(value_norm))
                if query_norm and query_norm in value_norm:
                    score += 6.0 if field in {"landmarks_identified", "final_place"} else 3.0
                    matched_fields.append(field)
                elif query_tokens.issubset(field_tokens):
                    score += 4.0 if field in {"landmarks_identified", "final_place"} else 2.0
                    matched_fields.append(field)
                else:
                    overlap = len(query_tokens & field_tokens)
                    if overlap:
                        score += overlap / max(len(query_tokens), 1)

            if score >= 1.0:
                scored.append(
                    {
                        "image_id": image_id,
                        "score": score,
                        "matched_fields": sorted(set(matched_fields)),
                        "final_place": row.get("final_place", ""),
                        "landmarks_identified": row.get("landmarks_identified", ""),
                    }
                )

        return sorted(scored, key=lambda item: item["score"], reverse=True)


class TextSearchVisualizer:
    def __init__(
        self,
        backend_name,
        es_host,
        es_index,
        redis_url,
        redis_key_prefix,
        image_root,
        device="cpu",
        vector_field="vector",
        image_id_field="image_id",
        cluster_id_field="cluster_id",
        model_id=None,
        model_version="c-radio_v4-h",
        lang_model="siglip2-g",
        cluster_weight=1.0,
        cls_weight=0.2,
        metadata_weight=0.3,
    ):
        self.es = Elasticsearch(es_host)
        self.redis = Redis.from_url(redis_url)
        self.image_root = image_root
        self.redis_key_prefix = redis_key_prefix
        self.es_index = es_index
        self.vector_field = vector_field
        self.image_id_field = image_id_field
        self.cluster_id_field = cluster_id_field
        self.embedding_type_field = "embedding_type"
        self.cluster_weight = float(cluster_weight)
        self.cls_weight = float(cls_weight)
        self.metadata_weight = float(metadata_weight)
        self.device = device

        print(f"Loading {backend_name} text encoder on {device}...")
        self.backend = create_backend(
            backend_name=backend_name,
            device=device,
            model_id=model_id,
            model_version=model_version,
            lang_model=lang_model,
        )

    @torch.no_grad()
    def encode_prompts(self, prompts):
        embeddings = self.backend.encode_text(prompts)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        return F.normalize(embeddings, dim=-1)

    @staticmethod
    def normalize_negative_prompts(negative_text):
        prompts = []
        if negative_text:
            if isinstance(negative_text, str):
                prompts = [part.strip() for part in negative_text.split(",") if part.strip()]
            elif isinstance(negative_text, list):
                prompts = [str(part).strip() for part in negative_text if str(part).strip()]

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
                "filter": self.cluster_doc_query(),
            },
            _source=[self.image_id_field, self.cluster_id_field],
            size=candidate_k,
        )
        return response["hits"]["hits"]

    def cluster_doc_query(self):
        return {
            "bool": {
                "should": [
                    {"term": {self.embedding_type_field: "cluster"}},
                    {"bool": {"must_not": {"exists": {"field": self.embedding_type_field}}}},
                ],
                "minimum_should_match": 1,
            }
        }

    def combine_with_cluster_filter(self, query):
        return {"bool": {"filter": [query, self.cluster_doc_query()]}}

    def score_candidate_query(self, positive_vector, negative_vectors, temperature, candidate_query, size):
        script_source = f"""
double pos = cosineSimilarity(params.positive_vector, '{self.vector_field}');
double numer = Math.exp(pos * params.temperature);
double denom = numer;
for (neg in params.negative_vectors) {{
  double negScore = cosineSimilarity(neg, '{self.vector_field}');
  denom += Math.exp(negScore * params.temperature);
}}
return numer / denom;
"""
        response = self.es.search(
            index=self.es_index,
            query={
                "script_score": {
                    "query": candidate_query,
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
            _source=[
                self.image_id_field,
                self.cluster_id_field,
                self.embedding_type_field,
                "final_place",
                "landmarks_identified",
            ],
            size=size,
        )
        return response["hits"]["hits"]

    def image_auxiliary_scores(self, image_ids, positive_vector, negative_vectors, temperature):
        if not image_ids or (self.cls_weight == 0.0 and self.metadata_weight == 0.0):
            return {}

        unique_image_ids = sorted(set(image_ids))
        query = {
            "bool": {
                "filter": [
                    {"terms": {self.image_id_field: unique_image_ids}},
                    {"terms": {self.embedding_type_field: ["cls", "metadata"]}},
                ]
            }
        }
        hits = self.score_candidate_query(
            positive_vector=positive_vector,
            negative_vectors=negative_vectors,
            temperature=temperature,
            candidate_query=query,
            size=max(len(unique_image_ids) * 2, 1),
        )
        scores = {}
        for hit in hits:
            source = hit.get("_source", {})
            image_id = source.get(self.image_id_field)
            embedding_type = source.get(self.embedding_type_field)
            if not image_id or embedding_type not in {"cls", "metadata"}:
                continue
            payload = scores.setdefault(image_id, {"cls_score": 0.0, "metadata_score": 0.0})
            key = "cls_score" if embedding_type == "cls" else "metadata_score"
            payload[key] = max(payload[key], float(hit.get("_score", 0.0)))
        return scores

    def apply_weighted_cluster_scores(self, scored_hits, auxiliary_scores):
        for item in scored_hits:
            aux = auxiliary_scores.get(item["image_id"], {})
            cluster_score = float(item["cluster_score"])
            cls_score = float(aux.get("cls_score", 0.0))
            metadata_score = float(aux.get("metadata_score", 0.0))
            item["cls_score"] = cls_score
            item["metadata_score"] = metadata_score
            item["score"] = (
                self.cluster_weight * cluster_score
                + self.cls_weight * cls_score
                + self.metadata_weight * metadata_score
            )
        return scored_hits

    def score_all_clusters_for_images(self, image_ids, query_text, negative_prompts, temperature=10.0):
        """Score every cluster in displayed images so visualization is a full-image heatmap."""
        unique_image_ids = sorted(set(image_ids))
        if not unique_image_ids:
            return {}

        prompts = [query_text] + self.normalize_negative_prompts(negative_prompts)
        text_vectors = self.encode_prompts(prompts)
        positive_vector = text_vectors[0].detach().cpu().numpy().tolist()
        negative_vectors = [vec.detach().cpu().numpy().tolist() for vec in text_vectors[1:]]

        hits = self.score_candidate_query(
            positive_vector=positive_vector,
            negative_vectors=negative_vectors,
            temperature=temperature,
            candidate_query=self.combine_with_cluster_filter({"terms": {self.image_id_field: unique_image_ids}}),
            size=max(len(unique_image_ids) * 64, 64),
        )

        scored_hits = []
        for hit in hits:
            source = hit.get("_source", {})
            scored_hits.append(
                {
                    "image_id": source[self.image_id_field],
                    "cluster_id": int(source.get(self.cluster_id_field, 0)),
                    "score": float(hit.get("_score", 0.0)),
                    "cluster_score": float(hit.get("_score", 0.0)),
                    "cls_score": 0.0,
                    "metadata_score": 0.0,
                    "raw_score": float(hit.get("_score", 0.0)),
                    "final_place": source.get("final_place", ""),
                    "landmarks_identified": source.get("landmarks_identified", ""),
                }
            )

        auxiliary_scores = self.image_auxiliary_scores(
            image_ids=unique_image_ids,
            positive_vector=positive_vector,
            negative_vectors=negative_vectors,
            temperature=temperature,
        )
        scored_hits = self.apply_weighted_cluster_scores(scored_hits, auxiliary_scores)

        grouped = {}
        for item in scored_hits:
            grouped.setdefault(item["image_id"], []).append(item)
        for items in grouped.values():
            items.sort(key=lambda item: item["score"], reverse=True)
        return grouped

    def direct_search_with_negatives(
        self,
        query_text,
        negative_text,
        candidate_k,
        top_k,
        temperature,
        result_mode="image",
        metadata_candidate_image_ids=None,
    ):
        negative_prompts = self.normalize_negative_prompts(negative_text)
        prompts = [query_text] + negative_prompts
        text_vectors = self.encode_prompts(prompts)

        positive_vector = text_vectors[0].detach().cpu().numpy().tolist()
        negative_vectors = [vec.detach().cpu().numpy().tolist() for vec in text_vectors[1:]]

        if metadata_candidate_image_ids:
            candidate_query = self.combine_with_cluster_filter({"terms": {self.image_id_field: metadata_candidate_image_ids}})
            hits = self.score_candidate_query(
                positive_vector=positive_vector,
                negative_vectors=negative_vectors,
                temperature=temperature,
                candidate_query=candidate_query,
                size=max(candidate_k, top_k * 10),
            )
        else:
            preselected_hits = self.knn_search_candidates(query_vector=positive_vector, candidate_k=candidate_k)
            candidate_ids = [hit["_id"] for hit in preselected_hits]
            if not candidate_ids:
                return [], negative_prompts
            hits = self.score_candidate_query(
                positive_vector=positive_vector,
                negative_vectors=negative_vectors,
                temperature=temperature,
                candidate_query=self.combine_with_cluster_filter({"ids": {"values": candidate_ids}}),
                size=candidate_k,
            )

        scored_hits = []
        for hit in hits:
            source = hit.get("_source", {})
            cluster_id = source.get(self.cluster_id_field, 0)
            scored_hits.append(
                {
                    "image_id": source[self.image_id_field],
                    "cluster_id": int(cluster_id),
                    "score": float(hit.get("_score", 0.0)),
                    "cluster_score": float(hit.get("_score", 0.0)),
                    "cls_score": 0.0,
                    "metadata_score": 0.0,
                    "raw_score": float(hit.get("_score", 0.0)),
                    "final_place": source.get("final_place", ""),
                    "landmarks_identified": source.get("landmarks_identified", ""),
                }
            )

        auxiliary_scores = self.image_auxiliary_scores(
            image_ids=[item["image_id"] for item in scored_hits],
            positive_vector=positive_vector,
            negative_vectors=negative_vectors,
            temperature=temperature,
        )
        scored_hits = self.apply_weighted_cluster_scores(scored_hits, auxiliary_scores)

        if result_mode == "cluster":
            results = sorted(scored_hits, key=lambda item: item["score"], reverse=True)
            return results[:top_k], negative_prompts

        best_per_image = {}
        for item in scored_hits:
            image_id = item["image_id"]
            if image_id not in best_per_image:
                best_per_image[image_id] = {
                    "image_id": image_id,
                    "score": item["score"],
                    "cluster_hits": [item],
                    "final_place": item.get("final_place", ""),
                    "landmarks_identified": item.get("landmarks_identified", ""),
                }
            else:
                best_per_image[image_id]["score"] = max(best_per_image[image_id]["score"], item["score"])
                best_per_image[image_id]["cluster_hits"].append(item)

        for payload in best_per_image.values():
            dedup = {}
            for hit in payload["cluster_hits"]:
                cluster_id = int(hit["cluster_id"])
                if cluster_id not in dedup or hit["score"] > dedup[cluster_id]["score"]:
                    dedup[cluster_id] = hit
            payload["cluster_hits"] = sorted(dedup.values(), key=lambda hit: hit["score"], reverse=True)

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
        return np.frombuffer(decoded, dtype=np.dtype(dtype_name)).reshape(height, width)

    def make_similarity_overlay(self, image, cluster_id_map, cluster_items, score_range=None, score_threshold=None):
        image_np = np.asarray(image).astype(np.float32) / 255.0
        score_map = np.zeros_like(cluster_id_map, dtype=np.float32)
        if not cluster_items:
            return image_np, score_map

        cluster_scores = {}
        for item in cluster_items:
            cluster_id = int(item["cluster_id"])
            score = float(item["score"])
            if score_threshold is not None and score < score_threshold:
                continue
            cluster_scores[cluster_id] = max(cluster_scores.get(cluster_id, 0.0), score)

        for cluster_id, score in cluster_scores.items():
            score_map[cluster_id_map == cluster_id] = score

        heatmap = cv2.resize(score_map, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
        if np.count_nonzero(heatmap) > 0:
            if score_range is None:
                positive = heatmap[heatmap > 0]
                low = float(positive.min())
                high = float(positive.max())
            else:
                low, high = score_range
            if high <= low:
                high = low + 1e-8
            heatmap = np.clip((heatmap - low) / max(high - low, 1e-8), 0.0, 1.0)
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=8, sigmaY=8)

        colored = plt.get_cmap("magma")(heatmap)[..., :3].astype(np.float32)
        alpha = (heatmap ** 0.8) * 0.75
        overlay = image_np * (1.0 - alpha[..., None]) + colored * alpha[..., None]
        return np.clip(overlay, 0.0, 1.0), heatmap

    def make_single_cluster_overlay(self, image, cluster_id_map, cluster_item, score_range=None):
        image_np = np.asarray(image).astype(np.float32) / 255.0
        cluster_id = int(cluster_item["cluster_id"])
        score = float(cluster_item.get("score", 0.0))

        mask = (cluster_id_map == cluster_id).astype(np.uint8)
        mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any():
            return image_np

        if score_range is None:
            normalized = 1.0
        else:
            low, high = score_range
            normalized = (score - low) / max(high - low, 1e-8)
            normalized = float(np.clip(normalized, 0.15, 1.0))

        color = np.asarray(plt.get_cmap("magma")(normalized)[:3], dtype=np.float32)
        overlay = image_np.copy()
        alpha = 0.62
        overlay[mask] = image_np[mask] * (1.0 - alpha) + color * alpha

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = mask.astype(np.uint8)
        boundary = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool) ^ cv2.erode(
            mask_u8, kernel, iterations=1
        ).astype(bool)
        overlay[boundary] = np.asarray([1.0, 0.95, 0.1], dtype=np.float32)
        return np.clip(overlay, 0.0, 1.0)

    def resolve_image_path(self, image_id):
        image_path = os.path.join(self.image_root, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path

    def visualize_results(
        self,
        results,
        query_text,
        negative_prompts,
        output_path=None,
        result_mode="image",
        temperature=10.0,
        heatmap_top_percent=35.0,
        heatmap_min_score=None,
    ):
        if not results:
            print("No matches found.")
            return

        if result_mode == "cluster":
            display_items = [
                (result["image_id"], [result], float(result.get("score", 0.0)))
                for result in results
            ]
        else:
            display_items = []
            for result in results:
                cluster_items = result.get("cluster_hits")
                if not cluster_items:
                    cluster_items = [
                        {
                            "image_id": result["image_id"],
                            "cluster_id": int(result.get("cluster_id", 0)),
                            "score": float(result.get("score", 0.0)),
                        }
                    ]
                display_items.append((result["image_id"], cluster_items, result["score"]))

        if result_mode != "cluster":
            full_heatmap_scores = self.score_all_clusters_for_images(
                image_ids=[image_id for image_id, _, _ in display_items],
                query_text=query_text,
                negative_prompts=negative_prompts,
                temperature=temperature,
            )
            display_items = [
                (image_id, full_heatmap_scores.get(image_id, image_results), image_score)
                for image_id, image_results, image_score in display_items
            ]
        global_scores = [
            float(item["score"])
            for _, image_results, _ in display_items
            for item in image_results
        ]
        if global_scores:
            global_score_range = (float(min(global_scores)), float(max(global_scores)))
            percentile_threshold = float(np.percentile(global_scores, 100.0 - heatmap_top_percent))
            if heatmap_min_score is None:
                score_threshold = percentile_threshold
            else:
                score_threshold = max(float(heatmap_min_score), percentile_threshold)
        else:
            global_score_range = None
            score_threshold = None

        cols = min(3, len(display_items))
        rows = math.ceil(len(display_items) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
        axes = np.atleast_1d(axes).reshape(rows, cols)

        for ax in axes.flat:
            ax.axis("off")

        for idx, (image_id, image_results, image_score) in enumerate(display_items):
            ax = axes[idx // cols, idx % cols]
            image_path = self.resolve_image_path(image_id)
            image = Image.open(image_path).convert("RGB")
            cluster_id_map = self.load_feature_map(image_id)
            if result_mode == "cluster":
                overlay = self.make_single_cluster_overlay(
                    image,
                    cluster_id_map,
                    image_results[0],
                    score_range=global_score_range,
                )
            else:
                overlay, _ = self.make_similarity_overlay(
                    image,
                    cluster_id_map,
                    image_results,
                    score_range=global_score_range,
                    score_threshold=score_threshold,
                )

            ax.imshow(overlay)
            neg_text = ", ".join(negative_prompts)
            cluster_text = ", ".join(
                f"{item['cluster_id']}:{item['score']:.4f}" for item in image_results[:6]
            )
            rank_text = f"rank={idx + 1} | " if result_mode == "cluster" else ""
            ax.set_title(
                f"{rank_text}{image_id}\n"
                f"image_score={image_score:.4f} | clusters={cluster_text}\n"
                f"query='{query_text}' vs [{neg_text}]"
                + ("" if result_mode == "cluster" else f" | global scale, top {heatmap_top_percent:g}%"),
                fontsize=11,
            )
            ax.axis("off")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test Elasticsearch text search with Redis-backed feature-map visualization.")
    parser.add_argument("query", type=str, help="Positive text query")
    parser.add_argument("--backend", type=str, default="tips", choices=["tips", "talk2dino", "radseg"])
    parser.add_argument("--model_id", type=str, default="google/tipsv2-b14")
    parser.add_argument("--negative_text", type=str, default="background", help="Comma-separated negative prompts")
    parser.add_argument("--top_k", type=int, default=6, help="Number of unique images to visualize")
    parser.add_argument("--candidate_k", type=int, default=120, help="Number of candidate clusters retrieved from ES before direct negative scoring")
    parser.add_argument("--temperature", type=float, default=10.0, help="Softmax temperature for reranking")
    parser.add_argument("--result_mode", type=str, default="image", choices=["image", "cluster"], help="Rank by best image or by cluster hits")
    parser.add_argument("--es_host", type=str, default="http://localhost:9200", help="Elasticsearch host")
    parser.add_argument("--es_index", type=str, default="tips_images", help="Elasticsearch index name")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0", help="Redis connection URL")
    parser.add_argument("--redis_key_prefix", type=str, default="tips_fm", help="Redis key prefix used for feature maps")
    parser.add_argument("--image_root", type=str, default="images", help="Directory containing original images")
    parser.add_argument("--metadata_csv", type=str, default="images_metadata.csv", help="Optional metadata CSV for specific landmark/place query filtering")
    parser.add_argument("--metadata_filter", type=str, default="auto", choices=["auto", "off", "force"], help="Use metadata filtering for specific queries")
    parser.add_argument("--metadata_max_images", type=int, default=500, help="Maximum metadata-matched images allowed into vector reranking")
    parser.add_argument("--model_version", type=str, default="c-radio_v4-h", help="RADSeg model version")
    parser.add_argument("--lang_model", type=str, default="siglip2-g", help="RADSeg language model")
    parser.add_argument("--device", type=str, default="cpu", help="Device for text encoding")
    parser.add_argument("--vector_field", type=str, default="vector", help="ES vector field name")
    parser.add_argument("--cluster_weight", type=float, default=1.0, help="Weight for each cluster's own text score")
    parser.add_argument("--cls_weight", type=float, default=0.2, help="Weight for the parent image CLS/global score")
    parser.add_argument("--metadata_weight", type=float, default=0.3, help="Weight for the parent image metadata score")
    parser.add_argument("--heatmap_top_percent", type=float, default=35.0, help="Only color clusters in the global top N percent of displayed cluster scores")
    parser.add_argument("--heatmap_min_score", type=float, default=None, help="Optional absolute minimum weighted score to color")
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to save the matplotlib figure")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = build_default_output_path(
            backend=args.backend,
            es_index=args.es_index,
            query_text=args.query,
            result_mode=args.result_mode,
        )
        print(f"Auto output path: {args.output_path}")

    visualizer = TextSearchVisualizer(
        backend_name=args.backend,
        es_host=args.es_host,
        es_index=args.es_index,
        redis_url=args.redis_url,
        redis_key_prefix=args.redis_key_prefix,
        image_root=args.image_root,
        model_version=args.model_version,
        lang_model=args.lang_model,
        device=args.device,
        vector_field=args.vector_field,
        model_id=args.model_id,
        cluster_weight=args.cluster_weight,
        cls_weight=args.cls_weight,
        metadata_weight=args.metadata_weight,
    )

    if not visualizer.es.ping():
        raise SystemExit(f"Could not connect to Elasticsearch at {args.es_host}")
    visualizer.redis.ping()

    router = MetadataQueryRouter(args.metadata_csv, max_images=args.metadata_max_images)
    route = router.route(args.query, mode=args.metadata_filter)
    print(
        f"Query route: {route['query_type']} "
        f"support={route['support']} reason={route['reason']}"
    )
    for example in route["examples"][:3]:
        print(
            "  metadata match: "
            f"image={example['image_id']} score={example['score']:.2f} "
            f"fields={','.join(example['matched_fields']) or '-'} "
            f"place={printable(example['final_place'])} landmark={printable(example['landmarks_identified'])}"
        )

    results, negative_prompts = visualizer.direct_search_with_negatives(
        query_text=args.query,
        negative_text=args.negative_text,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        temperature=args.temperature,
        result_mode=args.result_mode,
        metadata_candidate_image_ids=route["candidate_image_ids"],
    )

    print(f"Top {len(results)} results for '{args.query}':")
    for rank, result in enumerate(results, start=1):
        if args.result_mode == "cluster":
            print(
                f"{rank}. image={result['image_id']} cluster={result['cluster_id']} "
                f"score={result['score']:.4f} cluster={result.get('cluster_score', 0.0):.4f} "
                f"cls={result.get('cls_score', 0.0):.4f} metadata={result.get('metadata_score', 0.0):.4f} "
                f"raw_es={result['raw_score']:.4f} "
                f"place={printable(result.get('final_place', ''))}"
            )
        else:
            cluster_text = ", ".join(
                f"{item['cluster_id']}:{item['score']:.4f}"
                f"(c={item.get('cluster_score', 0.0):.3f},cls={item.get('cls_score', 0.0):.3f},m={item.get('metadata_score', 0.0):.3f})"
                for item in result.get("cluster_hits", [])[:6]
            )
            print(
                f"{rank}. image={result['image_id']} score={result['score']:.4f} "
                f"clusters=[{cluster_text}] place={printable(result.get('final_place', ''))}"
            )

    visualizer.visualize_results(
        results=results,
        query_text=args.query,
        negative_prompts=negative_prompts,
        output_path=args.output_path,
        result_mode=args.result_mode,
        temperature=args.temperature,
        heatmap_top_percent=args.heatmap_top_percent,
        heatmap_min_score=args.heatmap_min_score,
    )


if __name__ == "__main__":
    main()
