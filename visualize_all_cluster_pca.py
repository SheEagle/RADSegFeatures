import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from demo_talk2dino_v2_anyup_cluster_pca import (
    DEFAULT_ANYUP_MODEL,
    DEFAULT_IMAGE_PATH,
    compute_pca_rgb,
    load_talk2dino_model,
    upsample_anyup_features,
)
from demo_talk2dino_v2_anyup_cluster_search import (
    cluster_feature_map,
    load_epoc_segmenter,
    sam3_feature_map,
    subobject_feature_map,
    textregion_feature_map,
)


def resize_label_map(label_map, width, height):
    return cv2.resize(
        label_map.astype(np.int32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def pca_cluster_panel(pca_rgb, label_map, cluster_id, color, alpha):
    panel = pca_rgb.astype(np.float32).copy()
    mask = label_map == cluster_id
    panel[~mask] *= 0.18
    panel[mask] = panel[mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha

    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    panel[boundary] = 1.0
    return np.clip(panel, 0.0, 1.0)


def all_clusters_single_panel(image_np, pca_rgb, label_map, region_count, pca_alpha, boundary_width):
    base = image_np.astype(np.float32) / 255.0
    panel = base * (1.0 - pca_alpha) + pca_rgb.astype(np.float32) * pca_alpha
    boundaries = np.zeros(label_map.shape, dtype=bool)
    boundaries[:, 1:] |= label_map[:, 1:] != label_map[:, :-1]
    boundaries[1:, :] |= label_map[1:, :] != label_map[:-1, :]
    kernel_size = max(1, int(boundary_width))
    boundaries = cv2.dilate(
        boundaries.astype(np.uint8),
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1,
    ).astype(bool)
    panel[boundaries] = 1.0

    label_image = (np.clip(panel, 0.0, 1.0) * 255).astype(np.uint8)
    for cluster_id in range(region_count):
        ys, xs = np.where(label_map == cluster_id)
        if len(xs) == 0:
            continue
        cx = int(np.median(xs))
        cy = int(np.median(ys))
        cv2.putText(
            label_image,
            str(cluster_id),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            label_image,
            str(cluster_id),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return label_image


def get_regions(args, hr_features, image, device):
    if args.segmentation_mode == "epoc":
        segmenter = load_epoc_segmenter(
            repo_path=args.subobjects_repo,
            checkpoint=args.subobject_checkpoint,
            image_resolution=args.subobject_resolution,
            threshold=args.subobject_threshold,
            max_tokens=args.subobject_max_tokens,
            crop=args.subobject_crop,
            device=device,
        )
        return subobject_feature_map(hr_features, image, segmenter, min_mask_pixels=args.min_cluster_pixels)

    if args.segmentation_mode == "textregion":
        return textregion_feature_map(
            hr_features,
            image,
            repo_path=args.textregion_repo,
            model_cfg=args.textregion_model_cfg,
            checkpoint=args.textregion_sam2_checkpoint,
            points_per_side=args.textregion_points_per_side,
            max_masks=args.textregion_max_masks,
            min_mask_pixels=args.min_cluster_pixels,
            device=device,
        )

    if args.segmentation_mode == "sam3":
        return sam3_feature_map(
            hr_features,
            image,
            query=args.query,
            repo_path=args.sam3_repo,
            checkpoint_path=args.sam3_checkpoint_path,
            resolution=args.sam3_resolution,
            confidence_threshold=args.sam3_confidence_threshold,
            max_masks=args.sam3_max_masks,
            min_mask_pixels=args.min_cluster_pixels,
            device=device,
        )

    return cluster_feature_map(
        hr_features,
        num_clusters=args.num_clusters,
        min_cluster_pixels=args.min_cluster_pixels,
        merge_similarity=args.merge_similarity,
    )


def render(args):
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
    image_np = np.asarray(image)
    hr_features = upsample_anyup_features(
        model,
        upsampler,
        image,
        device,
        q_chunk_size=args.anyup_q_chunk_size,
        output_size=tuple(args.anyup_output_size) if args.anyup_output_size else None,
    )
    pca_rgb = compute_pca_rgb(hr_features)
    pca_image = cv2.resize(
        pca_rgb.astype(np.float32),
        (image.width, image.height),
        interpolation=cv2.INTER_LINEAR,
    )

    centers, label_map_t = get_regions(args, hr_features, image, device)
    label_map = resize_label_map(label_map_t.detach().cpu().numpy(), image.width, image.height)
    region_count = int(centers.shape[0])
    print(f"regions={region_count}")

    if args.layout == "single":
        panel = all_clusters_single_panel(
            image_np,
            pca_image,
            label_map,
            region_count,
            pca_alpha=args.pca_alpha,
            boundary_width=args.boundary_width,
        )
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(panel)
        ax.set_title(
            f"All clusters on PCA | {args.segmentation_mode} | regions={region_count}\n{image_path.name}",
            fontsize=14,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved all-cluster PCA image to: {output_path}")
        return

    cols = min(args.cols, max(1, region_count))
    rows = math.ceil(region_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows), squeeze=False)
    palette = plt.get_cmap("tab20")

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= region_count:
            continue
        mask = label_map == idx
        area = int(mask.sum())
        pct = 100.0 * area / float(label_map.size)
        panel = pca_cluster_panel(
            pca_image,
            label_map,
            idx,
            color=palette(idx % 20)[:3],
            alpha=args.alpha,
        )
        ax.imshow(panel)
        ax.set_title(f"cluster {idx} | {area} px ({pct:.1f}%)", fontsize=10)

    fig.suptitle(
        f"All cluster PCA subplots | {args.segmentation_mode} | regions={region_count}\n{image_path.name}",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved all-cluster PCA grid to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Render PCA visualization for every cluster/region.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model_id", default="lorebianchi98/Talk2DINOv3-ViTL")
    parser.add_argument("--output_path", default="scratch/all_cluster_pca_grid.png")
    parser.add_argument("--segmentation_mode", choices=["kmeans", "epoc", "textregion", "sam3"], default="kmeans")
    parser.add_argument("--layout", choices=["single", "grid"], default="single")
    parser.add_argument("--query", default="river")
    parser.add_argument("--num_clusters", type=int, default=15)
    parser.add_argument("--min_cluster_pixels", type=int, default=8)
    parser.add_argument("--merge_similarity", type=float, default=0.97)
    parser.add_argument("--subobjects_repo", default="scratch/subobjects_repo")
    parser.add_argument("--subobject_checkpoint", default="chendelong/DirectSAM-b0-1024px-sa1b-2ep-1017")
    parser.add_argument("--subobject_resolution", type=int, default=1024)
    parser.add_argument("--subobject_threshold", type=float, default=0.1)
    parser.add_argument("--subobject_max_tokens", type=int, default=32)
    parser.add_argument("--subobject_crop", type=int, default=1)
    parser.add_argument("--textregion_repo", default="scratch/TextRegion")
    parser.add_argument("--textregion_sam2_checkpoint", default="scratch/TextRegion/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--textregion_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--textregion_points_per_side", type=int, default=16)
    parser.add_argument("--textregion_max_masks", type=int, default=32)
    parser.add_argument("--sam3_repo", default="scratch/sam3")
    parser.add_argument("--sam3_checkpoint_path", default="sam3.pt")
    parser.add_argument("--sam3_resolution", type=int, default=1008)
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.25)
    parser.add_argument("--sam3_max_masks", type=int, default=32)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--pca_alpha", type=float, default=0.35)
    parser.add_argument("--boundary_width", type=int, default=2)
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL)
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=10)
    parser.add_argument("--anyup_output_size", nargs=2, type=int, default=[384, 384], metavar=("WIDTH", "HEIGHT"))
    return parser


if __name__ == "__main__":
    render(build_parser().parse_args())
