import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from demo_talk2dino_v2_anyup_cluster_pca import (
    DEFAULT_ANYUP_MODEL,
    DEFAULT_IMAGE_PATH,
    build_palette,
    compute_pca_rgb,
    load_talk2dino_model,
    upsample_anyup_features,
)
from demo_talk2dino_v2_anyup_cluster_search import textregion_feature_map
from visualize_epoc_pca import boundary_overlay, resize_overlay


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

    print("Segmenting with TextRegion/SAM2...")
    centers, label_map_t = textregion_feature_map(
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
    label_map = label_map_t.detach().cpu().numpy()
    region_count = int(centers.shape[0])
    palette = build_palette(region_count)
    segmentation_rgb = palette[label_map]
    segmentation_overlay = resize_overlay(image, segmentation_rgb, alpha=args.overlay_alpha)
    boundaries = boundary_overlay(image, label_map)

    panels = [
        (np.asarray(image), "Original"),
        (pca_image, f"AnyUp PCA\n{hr_features.shape[-2]}x{hr_features.shape[-1]}"),
        (segmentation_overlay, f"TextRegion/SAM2 segmentation\nmax_masks={args.textregion_max_masks}, regions={region_count}"),
        (boundaries, "TextRegion/SAM2 boundaries"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(f"Talk2DINOv3 + AnyUp PCA vs TextRegion/SAM2 masks\n{image_path.name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize TextRegion/SAM2 segmentation beside AnyUp PCA.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model_id", default="lorebianchi98/Talk2DINOv3-ViTL")
    parser.add_argument("--output_path", default="scratch/textregion_segmentation_pca.png")
    parser.add_argument("--textregion_repo", default="scratch/TextRegion")
    parser.add_argument("--textregion_sam2_checkpoint", default="scratch/TextRegion/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--textregion_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--textregion_points_per_side", type=int, default=16)
    parser.add_argument("--textregion_max_masks", type=int, default=32)
    parser.add_argument("--min_cluster_pixels", type=int, default=8)
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL)
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=10)
    parser.add_argument("--anyup_output_size", nargs=2, type=int, default=[384, 384], metavar=("WIDTH", "HEIGHT"))
    return parser


if __name__ == "__main__":
    render(build_parser().parse_args())
