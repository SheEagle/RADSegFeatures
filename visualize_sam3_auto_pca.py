import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import SAM

from demo_talk2dino_v2_anyup_cluster_pca import (
    DEFAULT_ANYUP_MODEL,
    DEFAULT_IMAGE_PATH,
    compute_pca_rgb,
    load_talk2dino_model,
    upsample_anyup_features,
)
from demo_talk2dino_v2_anyup_cluster_search import feature_regions_from_masks
from visualize_all_cluster_pca import all_clusters_single_panel, resize_label_map


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

    print(f"Running Ultralytics SAM3 automatic segmentation: {args.sam3_checkpoint_path}")
    sam = SAM(args.sam3_checkpoint_path)
    results = sam.predict(
        str(image_path),
        device=0 if device == "cuda" else "cpu",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
    )
    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        raise RuntimeError("SAM3 automatic segmentation produced no masks.")

    masks = result.masks.data.detach().cpu().bool()
    print(f"raw_masks={masks.shape[0]}")

    centers, label_map_t = feature_regions_from_masks(
        hr_features,
        masks,
        min_mask_pixels=args.min_cluster_pixels,
        max_masks=args.max_masks,
        assign_uncovered=False,
    )

    pca_rgb = compute_pca_rgb(hr_features)
    pca_image = cv2.resize(
        pca_rgb.astype(np.float32),
        (image.width, image.height),
        interpolation=cv2.INTER_LINEAR,
    )
    label_map = resize_label_map(label_map_t.detach().cpu().numpy(), image.width, image.height)
    region_count = int(centers.shape[0])
    print(f"regions={region_count}")

    panel = all_clusters_single_panel(
        image_np,
        pca_image,
        label_map,
        region_count,
        pca_alpha=args.pca_alpha,
        boundary_width=args.boundary_width,
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(panel)
    ax.set_title(f"SAM3 automatic masks on PCA | regions={region_count}\n{image_path.name}", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SAM3 automatic PCA image to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize Ultralytics SAM3 automatic segmentation over AnyUp PCA.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model_id", default="lorebianchi98/Talk2DINOv3-ViTL")
    parser.add_argument("--output_path", default="scratch/all_cluster_pca_single_sam3_auto_light.png")
    parser.add_argument("--sam3_checkpoint_path", default="sam3.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_masks", type=int, default=64)
    parser.add_argument("--min_cluster_pixels", type=int, default=8)
    parser.add_argument("--pca_alpha", type=float, default=0.28)
    parser.add_argument("--boundary_width", type=int, default=2)
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL)
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=10)
    parser.add_argument("--anyup_output_size", nargs=2, type=int, default=[384, 384], metavar=("WIDTH", "HEIGHT"))
    return parser


if __name__ == "__main__":
    render(build_parser().parse_args())
