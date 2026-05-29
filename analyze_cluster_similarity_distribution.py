import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(description="Plot pairwise cosine similarity distribution among per-image cluster representatives.")
    parser.add_argument("--jsonl", default="features_talk2dinov3_anyup_k10_cls_metadata_1_1000.jsonl")
    parser.add_argument("--output_path", default="scratch/cluster_pairwise_similarity_distribution.png")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.90, 0.93, 0.95, 0.97])
    return parser


def iter_cluster_matrices(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            vectors = [cluster["v"] for cluster in record.get("clusters", [])]
            if len(vectors) < 2:
                continue
            yield record.get("image_id", ""), np.asarray(vectors, dtype=np.float32)


def main():
    args = build_parser().parse_args()
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    similarities = []
    per_image_max = []
    cluster_counts = []
    for _, vectors in iter_cluster_matrices(jsonl_path):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-8)
        sim = vectors @ vectors.T
        upper = sim[np.triu_indices(len(vectors), k=1)]
        similarities.extend(upper.tolist())
        per_image_max.append(float(upper.max()))
        cluster_counts.append(len(vectors))

    similarities = np.asarray(similarities, dtype=np.float32)
    per_image_max = np.asarray(per_image_max, dtype=np.float32)

    print(f"Images analyzed: {len(cluster_counts)}")
    print(f"Pair count: {len(similarities)}")
    print(f"Mean clusters/image: {np.mean(cluster_counts):.2f}")
    for value in [50, 75, 90, 95, 97, 99]:
        print(f"pair sim p{value}: {np.percentile(similarities, value):.4f}")
    print(f"per-image max sim mean: {per_image_max.mean():.4f}")
    for threshold in args.thresholds:
        pct_pairs = float((similarities >= threshold).mean() * 100.0)
        pct_images = float((per_image_max >= threshold).mean() * 100.0)
        print(f">= {threshold:.2f}: pairs={pct_pairs:.2f}% images_with_any_pair={pct_images:.2f}%")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(similarities, bins=70, color="#4f8fc0", alpha=0.85)
    for threshold in args.thresholds:
        ax.axvline(threshold, linestyle="--", linewidth=1.8, label=f"{threshold:.2f}")
    ax.set_title("Pairwise cosine similarity among cluster representatives")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Cluster-pair count")
    ax.legend(title="merge threshold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
