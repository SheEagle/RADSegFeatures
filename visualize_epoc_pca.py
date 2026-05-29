import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from demo_talk2dino_v2_anyup_cluster_pca import (
    DEFAULT_ANYUP_MODEL,
    DEFAULT_IMAGE_PATH,
    build_palette,
    compute_pca_rgb,
    load_talk2dino_model,
    upsample_anyup_features,
)
from demo_talk2dino_v2_anyup_cluster_search import load_epoc_segmenter, subobject_feature_map


def resize_overlay(image, overlay_rgb, alpha=0.55, interpolation=cv2.INTER_NEAREST):
    overlay = cv2.resize(overlay_rgb, (image.width, image.height), interpolation=interpolation)
    base = np.asarray(image).astype(np.float32) / 255.0
    overlay = overlay.astype(np.float32) / 255.0
    return np.clip(base * (1.0 - alpha) + overlay * alpha, 0.0, 1.0)


def boundary_overlay(image, label_map):
    labels = cv2.resize(
        label_map.astype(np.int32),
        (image.width, image.height),
        interpolation=cv2.INTER_NEAREST,
    )
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundaries = cv2.dilate(boundaries.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    base = np.asarray(image).astype(np.float32) / 255.0
    base[boundaries] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return base


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
        (pca_rgb * 255).astype(np.uint8),
        (image.width, image.height),
        interpolation=cv2.INTER_LINEAR,
    )

    rows = len(args.subobject_max_tokens)
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5.2 * rows), squeeze=False)

    for row_idx, max_tokens in enumerate(args.subobject_max_tokens):
        print(f"Segmenting with EPOC max_tokens={max_tokens}...")
        segmenter = load_epoc_segmenter(
            repo_path=args.subobjects_repo,
            checkpoint=args.subobject_checkpoint,
            image_resolution=args.subobject_resolution,
            threshold=args.subobject_threshold,
            max_tokens=max_tokens,
            crop=args.subobject_crop,
            device=device,
        )
        centers, label_map_t = subobject_feature_map(
            hr_features,
            image,
            segmenter,
            min_mask_pixels=args.min_cluster_pixels,
        )
        label_map = label_map_t.detach().cpu().numpy()
        region_count = int(centers.shape[0])
        palette = build_palette(region_count)
        segmentation_rgb = palette[label_map]
        segmentation_overlay = resize_overlay(image, segmentation_rgb, alpha=args.overlay_alpha)
        boundaries = boundary_overlay(image, label_map)

        panels = [
            (np.asarray(image), "Original"),
            (pca_image, f"AnyUp PCA\n{hr_features.shape[-2]}x{hr_features.shape[-1]}"),
            (segmentation_overlay, f"EPOC segmentation\nmax_tokens={max_tokens}, regions={region_count}"),
            (boundaries, "EPOC boundaries"),
        ]
        for col_idx, (panel, title) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

    fig.suptitle(f"Talk2DINOv3 + AnyUp PCA vs EPOC subobjects\n{image_path.name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize EPOC segmentation beside AnyUp PCA.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model_id", default="lorebianchi98/Talk2DINOv3-ViTL")
    parser.add_argument("--output_path", default="scratch/epoc_segmentation_pca_compare.png")
    parser.add_argument("--subobject_max_tokens", nargs="+", type=int, default=[10, 20, 32])
    parser.add_argument("--subobjects_repo", default="scratch/subobjects_repo")
    parser.add_argument("--subobject_checkpoint", default="chendelong/DirectSAM-b0-1024px-sa1b-2ep-1017")
    parser.add_argument("--subobject_resolution", type=int, default=1024)
    parser.add_argument("--subobject_threshold", type=float, default=0.1)
    parser.add_argument("--subobject_crop", type=int, default=1)
    parser.add_argument("--min_cluster_pixels", type=int, default=8)
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL)
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=10)
    parser.add_argument("--anyup_output_size", nargs=2, type=int, default=[384, 384], metavar=("WIDTH", "HEIGHT"))
    return parser


if __name__ == "__main__":
    render(build_parser().parse_args())
