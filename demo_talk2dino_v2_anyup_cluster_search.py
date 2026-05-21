import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from PIL import Image

from batch_extract_features import adaptive_spherical_kmeans
from demo_talk2dino_v2_anyup_cluster_pca import (
    DEFAULT_ANYUP_MODEL,
    DEFAULT_IMAGE_PATH,
    DEFAULT_MODEL_ID,
    load_talk2dino_model,
    upsample_anyup_features,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Single-image Talk2DINO v2 + AnyUp cluster search with RADSeg-style heatmap visualization."
    )
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="HF model id.")
    parser.add_argument("--query", default="river", help="Positive query text.")
    parser.add_argument("--negative_text", default="background", help="Comma-separated negative prompts.")
    parser.add_argument("--output_path", default=None, help="Output figure path.")
    parser.add_argument("--num_clusters", type=int, default=10, help="Maximum number of clusters.")
    parser.add_argument("--min_cluster_pixels", type=int, default=8, help="Minimum cluster size before reassignment.")
    parser.add_argument("--merge_similarity", type=float, default=0.95, help="Merge clusters above this cosine similarity.")
    parser.add_argument("--temperature", type=float, default=10.0, help="Contrastive softmax temperature.")
    parser.add_argument("--top_clusters", type=int, default=6, help="How many cluster hits to print in the title.")
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL, help="torch.hub AnyUp entrypoint.")
    parser.add_argument("--anyup_use_natten", action="store_true", help="Use NATTEN-based AnyUp model.")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=None, help="Optional AnyUp q_chunk_size.")
    parser.add_argument(
        "--anyup_output_size",
        nargs=2,
        type=int,
        default=[384, 384],
        metavar=("WIDTH", "HEIGHT"),
        help="AnyUp output size, default 384 384 for practical speed.",
    )
    return parser


def normalize_negative_prompts(negative_text):
    if not negative_text:
        return ["background"]
    prompts = [part.strip() for part in str(negative_text).split(",") if part.strip()]
    return prompts or ["background"]


@torch.no_grad()
def encode_prompts(model, prompts, device):
    embeddings = model.encode_text(prompts)
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    embeddings = embeddings.to(device)
    return F.normalize(embeddings, dim=-1)


def cluster_feature_map(feature_map: torch.Tensor, num_clusters: int, min_cluster_pixels: int, merge_similarity: float):
    b, c, h, w = feature_map.shape
    assert b == 1
    flat = feature_map[0].permute(1, 2, 0).reshape(-1, c)
    centers, labels = adaptive_spherical_kmeans(
        flat,
        max_clusters=num_clusters,
        min_cluster_pixels=min_cluster_pixels,
        merge_similarity=merge_similarity,
    )
    label_map = labels.reshape(h, w)
    return centers, label_map


def score_clusters(centers, positive_vector, negative_vectors, temperature):
    centers = F.normalize(centers, dim=-1)
    pos = torch.matmul(centers, positive_vector)
    numer = torch.exp(pos * temperature)
    denom = numer.clone()
    neg_scores = []
    for neg in negative_vectors:
        neg_score = torch.matmul(centers, neg)
        neg_scores.append(neg_score)
        denom = denom + torch.exp(neg_score * temperature)
    scores = numer / denom.clamp_min(1e-8)
    return pos, scores


def make_similarity_overlay(image, cluster_id_map, cluster_items):
    image_np = np.asarray(image).astype(np.float32) / 255.0
    score_map = np.zeros_like(cluster_id_map, dtype=np.float32)

    cluster_scores = {}
    for item in cluster_items:
        cluster_id = int(item["cluster_id"])
        cluster_scores[cluster_id] = max(cluster_scores.get(cluster_id, 0.0), float(item["score"]))

    for cluster_id, score in cluster_scores.items():
        score_map[cluster_id_map == cluster_id] = score

    heatmap = cv2.resize(score_map, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
    if np.count_nonzero(heatmap) > 0:
        positive = heatmap[heatmap > 0]
        low = float(np.percentile(positive, 5))
        high = float(np.percentile(positive, 95))
        if high <= low:
            low = float(positive.min())
            high = float(positive.max())
        heatmap = np.clip((heatmap - low) / max(high - low, 1e-8), 0.0, 1.0)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=8, sigmaY=8)

    colored = plt.get_cmap("magma")(heatmap)[..., :3].astype(np.float32)
    alpha = (heatmap ** 0.8) * 0.75
    overlay = image_np * (1.0 - alpha[..., None]) + colored * alpha[..., None]
    return np.clip(overlay, 0.0, 1.0), heatmap


def make_cluster_legend_overlay(image, cluster_id_map, cluster_items, top_clusters):
    image_np = np.asarray(image).astype(np.float32) / 255.0
    top_items = cluster_items[:top_clusters]
    if not top_items:
        return image_np, []

    if cluster_id_map.shape[:2] != image_np.shape[:2]:
        resized_cluster_map = cv2.resize(
            cluster_id_map.astype(np.int32),
            (image.width, image.height),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized_cluster_map = cluster_id_map

    palette = plt.get_cmap("tab10")
    overlay = image_np.copy()
    legend_entries = []
    positive_scores = np.array([max(float(item["score"]), 0.0) for item in top_items], dtype=np.float32)
    max_score = float(positive_scores.max()) if positive_scores.size else 1.0

    for idx, item in enumerate(top_items):
        cluster_id = int(item["cluster_id"])
        score = float(item["score"])
        mask = resized_cluster_map == cluster_id
        if not np.any(mask):
            continue

        color = np.array(palette(idx % 10)[:3], dtype=np.float32)
        alpha = 0.25 + 0.45 * (max(score, 0.0) / max(max_score, 1e-8))
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
        legend_entries.append(
            {
                "cluster_id": cluster_id,
                "score": score,
                "color": color,
            }
        )

    return np.clip(overlay, 0.0, 1.0), legend_entries


def render_demo(
    image_path: Path,
    model_id: str,
    query: str,
    negative_text: str,
    output_path: Path,
    num_clusters: int,
    min_cluster_pixels: int,
    merge_similarity: float,
    temperature: float,
    top_clusters: int,
    anyup_entrypoint: str,
    anyup_use_natten: bool,
    anyup_q_chunk_size,
    anyup_output_size,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {model_id}")
    model = load_talk2dino_model(model_id, device)

    print(f"Loading AnyUp: {anyup_entrypoint} (use_natten={anyup_use_natten})")
    upsampler = torch.hub.load(
        "wimmerth/anyup",
        anyup_entrypoint,
        use_natten=anyup_use_natten,
    ).to(device).eval()

    image = Image.open(image_path).convert("RGB")
    hr_features = upsample_anyup_features(
        model,
        upsampler,
        image,
        device,
        q_chunk_size=anyup_q_chunk_size,
        output_size=tuple(anyup_output_size) if anyup_output_size is not None else None,
    )

    centers, label_map = cluster_feature_map(
        hr_features,
        num_clusters=num_clusters,
        min_cluster_pixels=min_cluster_pixels,
        merge_similarity=merge_similarity,
    )

    negative_prompts = normalize_negative_prompts(negative_text)
    text_vectors = encode_prompts(model, [query] + negative_prompts, device)
    positive_vector = text_vectors[0]
    negative_vectors = [vec for vec in text_vectors[1:]]

    pos_scores, scores = score_clusters(centers, positive_vector, negative_vectors, temperature)
    cluster_items = []
    for cluster_id in range(centers.shape[0]):
        cluster_items.append(
            {
                "cluster_id": int(cluster_id),
                "score": float(scores[cluster_id].item()),
                "pos": float(pos_scores[cluster_id].item()),
            }
        )
    cluster_items.sort(key=lambda item: item["score"], reverse=True)

    overlay, heatmap = make_similarity_overlay(image, label_map.detach().cpu().numpy(), cluster_items)
    cluster_overlay, legend_entries = make_cluster_legend_overlay(
        image,
        label_map.detach().cpu().numpy(),
        cluster_items,
        top_clusters=top_clusters,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 6),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.55]},
    )

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="magma")
    axes[1].set_title(f"Cluster score map\nquery='{query}' vs [{', '.join(negative_prompts)}]")
    axes[1].axis("off")

    axes[2].imshow(cluster_overlay)
    cluster_text = ", ".join(
        f"{item['cluster_id']}:{item['score']:.4f}" for item in cluster_items[:top_clusters]
    )
    axes[2].set_title(
        f"Top cluster overlay + legend\n"
        f"clusters={cluster_text}\n"
        f"K={num_clusters}, final={centers.shape[0]}, merge={merge_similarity}"
    )
    axes[2].axis("off")

    axes[3].axis("off")
    axes[3].set_title("Legend")

    if legend_entries:
        handles = [
            Patch(
                facecolor=entry["color"],
                edgecolor="none",
                label=f"cluster {entry['cluster_id']} ({entry['score']:.3f})",
            )
            for entry in legend_entries
        ]
        axes[3].legend(
            handles=handles,
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path("scratch") / f"{image_path.stem}_talk2dino_v2_anyup_cluster_search_{args.query}.png"

    render_demo(
        image_path=image_path,
        model_id=args.model_id,
        query=args.query,
        negative_text=args.negative_text,
        output_path=output_path,
        num_clusters=args.num_clusters,
        min_cluster_pixels=args.min_cluster_pixels,
        merge_similarity=args.merge_similarity,
        temperature=args.temperature,
        top_clusters=args.top_clusters,
        anyup_entrypoint=args.anyup_entrypoint,
        anyup_use_natten=args.anyup_use_natten,
        anyup_q_chunk_size=args.anyup_q_chunk_size,
        anyup_output_size=args.anyup_output_size,
    )


if __name__ == "__main__":
    main()
