import argparse
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from compare_talk2dinov3_anyup_cluster_k import (
    DEFAULT_IMAGE_PATH,
    DEFAULT_MODEL_ID,
    draw_panel,
    overlay_to_image,
    pca_to_image,
)
from demo_talk2dino_v2_anyup_cluster_pca import (
    build_palette,
    cluster_feature_map,
    compute_pca_rgb,
    upsample_anyup_features,
)
from demo_talk2dino_v2_single_image import DEFAULT_ANYUP_MODEL, load_talk2dino_model


def build_parser():
    parser = argparse.ArgumentParser(description="Compare KMeans and HDBSCAN on Talk2DINOv3+AnyUp features.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_path", default="scratch/talk2dinov3_anyup_kmeans_vs_hdbscan.png")
    parser.add_argument("--kmeans_clusters", type=int, default=15)
    parser.add_argument("--min_cluster_pixels", type=int, default=16)
    parser.add_argument("--merge_similarity", type=float, default=0.97)
    parser.add_argument("--representative_mode", choices=["mean", "geometric_median"], default="mean")
    parser.add_argument("--hdbscan_sizes", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--hdbscan_min_samples", type=int, default=10)
    parser.add_argument("--hdbscan_feature_size", nargs=2, type=int, default=[128, 128], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL)
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=10)
    parser.add_argument("--anyup_output_size", nargs=2, type=int, default=[384, 384], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--panel_width", type=int, default=640)
    return parser


def hdbscan_feature_clustering(feature_map, min_cluster_size, min_samples, feature_size):
    pooled = F.interpolate(
        feature_map.float(),
        size=(int(feature_size[1]), int(feature_size[0])),
        mode="bilinear",
        align_corners=False,
    )
    pooled = F.normalize(pooled, dim=1)
    _, channels, height, width = pooled.shape
    flat = pooled[0].permute(1, 2, 0).reshape(-1, channels).detach().cpu().numpy()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
    )
    labels_np = clusterer.fit_predict(flat).reshape(height, width)
    valid_labels = sorted(label for label in np.unique(labels_np) if label >= 0)

    # Keep noise visually separate. Representatives are only computed for non-noise clusters.
    remapped = np.full_like(labels_np, fill_value=0, dtype=np.int64)
    for new_id, old_id in enumerate(valid_labels, start=1):
        remapped[labels_np == old_id] = new_id

    labels = torch.from_numpy(remapped.reshape(-1)).to(feature_map.device)
    full_labels = F.interpolate(
        torch.from_numpy(remapped).float().unsqueeze(0).unsqueeze(0),
        size=feature_map.shape[-2:],
        mode="nearest",
    )[0, 0].long().to(feature_map.device)

    dense_flat = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
    dense_flat = F.normalize(dense_flat.float(), dim=-1)
    up_labels = full_labels.reshape(-1)

    centers = []
    for cluster_id in range(1, len(valid_labels) + 1):
        mask = up_labels == cluster_id
        if mask.any():
            center = F.normalize(dense_flat[mask].mean(dim=0, keepdim=True), dim=-1)[0]
            centers.append(center)
    centers = torch.stack(centers, dim=0) if centers else dense_flat[:0]
    return centers, full_labels, int((labels == 0).sum().item()), len(valid_labels)


def render_overlay(image, centers, label_map, noise_id=0):
    labels_np = label_map.detach().cpu().numpy()
    if centers.shape[0] == 0:
        cluster_rgb = np.zeros((*labels_np.shape, 3), dtype=np.uint8)
        return overlay_to_image(image, cluster_rgb, alpha=0.0)

    palette = build_palette(int(centers.shape[0]) + 1)
    palette[noise_id] = np.array([20, 20, 20], dtype=np.uint8)
    cluster_rgb = palette[labels_np]
    return overlay_to_image(image, cluster_rgb, alpha=0.54)


def render_comparison(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = Path(args.image_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model: {args.model_id}")
    model = load_talk2dino_model(args.model_id, device)

    print(f"Loading AnyUp: {args.anyup_entrypoint} (use_natten={args.anyup_use_natten})")
    upsampler = torch.hub.load(
        "wimmerth/anyup",
        args.anyup_entrypoint,
        use_natten=args.anyup_use_natten,
    ).to(device).eval()

    image = Image.open(image_path).convert("RGB")
    canvas_size = (args.panel_width, max(1, int(args.panel_width * image.height / image.width)))

    print("Computing AnyUp feature map...")
    hr_features = upsample_anyup_features(
        model,
        upsampler,
        image,
        device,
        q_chunk_size=args.anyup_q_chunk_size,
        output_size=tuple(args.anyup_output_size) if args.anyup_output_size else None,
    )
    pca_rgb = compute_pca_rgb(hr_features)

    panels = [
        ("Original", image),
        (f"AnyUp PCA\n{hr_features.shape[-2]}x{hr_features.shape[-1]} feature map", pca_to_image(image, pca_rgb)),
    ]

    print("Clustering KMeans...")
    centers, label_map = cluster_feature_map(
        hr_features,
        num_clusters=args.kmeans_clusters,
        min_cluster_pixels=args.min_cluster_pixels,
        merge_similarity=args.merge_similarity,
        representative_mode=args.representative_mode,
    )
    panels.append(
        (
            f"KMeans K={args.kmeans_clusters}, final={centers.shape[0]}\nmerge={args.merge_similarity}",
            render_overlay(image, centers, label_map, noise_id=-1),
        )
    )

    for min_size in args.hdbscan_sizes:
        print(f"Clustering HDBSCAN min_cluster_size={min_size}...")
        centers, label_map, noise_tokens, num_clusters = hdbscan_feature_clustering(
            hr_features,
            min_cluster_size=min_size,
            min_samples=args.hdbscan_min_samples,
            feature_size=args.hdbscan_feature_size,
        )
        panels.append(
            (
                f"HDBSCAN min_size={min_size}, clusters={num_clusters}\n"
                f"min_samples={args.hdbscan_min_samples}, noise={noise_tokens}",
                render_overlay(image, centers, label_map),
            )
        )

    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, panel_image) in zip(axes.flat, panels):
        draw_panel(ax, panel_image, title, canvas_size)

    fig.suptitle(f"Talk2DINOv3 + AnyUp: KMeans vs HDBSCAN\n{image_path.name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to: {output_path}")


def main():
    render_comparison(build_parser().parse_args())


if __name__ == "__main__":
    main()
