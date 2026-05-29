import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from demo_talk2dino_v2_anyup_cluster_pca import DEFAULT_IMAGE_PATH
from demo_talk2dino_v2_anyup_cluster_search import load_textregion_mask_generator


@torch.no_grad()
def generate_textregion_masks(args, image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_textregion_mask_generator(
        repo_path=args.textregion_repo,
        model_cfg=args.textregion_model_cfg,
        checkpoint=args.textregion_sam2_checkpoint,
        points_per_side=args.textregion_points_per_side,
        device=device,
    )
    image_np = np.array(image.convert("RGB"), copy=True)
    image_tensor = torch.from_numpy(image_np).to(device=device, dtype=torch.float32)
    image_tensor_for_sam2 = torch.stack([image_tensor])
    image_tensor_for_sam2 = generator.predictor._transforms(image_tensor_for_sam2)
    ori_shape = image_np.shape[:2]
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device == "cuda"):
        sam2_masks = generator.generate_for_batch(image_tensor_for_sam2, [ori_shape], None)

    masks = torch.stack([mask["segmentations"] for mask in sam2_masks[0]]).bool()
    areas = masks.flatten(1).sum(dim=1)
    keep = areas >= max(1, args.min_mask_pixels)
    masks = masks[keep]
    areas = areas[keep]
    order = torch.argsort(areas, descending=True)
    if args.max_regions and args.max_regions > 0:
        order = order[: args.max_regions]
    return masks[order].cpu().numpy(), areas[order].cpu().numpy()


def highlight_region(image_np, mask, color, alpha):
    overlay = image_np.astype(np.float32) / 255.0
    color = np.array(color, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    overlay[boundary] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return np.clip(overlay, 0.0, 1.0)


def render(args):
    image_path = Path(args.image_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    masks, areas = generate_textregion_masks(args, image)
    if len(masks) == 0:
        raise RuntimeError("TextRegion/SAM2 produced no masks after filtering.")

    cols = min(args.cols, len(masks))
    rows = math.ceil(len(masks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    palette = plt.get_cmap("tab20")

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= len(masks):
            continue
        color = palette(idx % 20)[:3]
        panel = highlight_region(image_np, masks[idx], color=color, alpha=args.alpha)
        ax.imshow(panel)
        pct = 100.0 * float(areas[idx]) / float(image_np.shape[0] * image_np.shape[1])
        ax.set_title(f"region {idx} | area={int(areas[idx])} ({pct:.1f}%)", fontsize=10)

    fig.suptitle(f"TextRegion/SAM2 individual regions\n{image_path.name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved region grid to: {output_path}")
    print(f"regions={len(masks)}")


def build_parser():
    parser = argparse.ArgumentParser(description="Render one highlighted subplot per TextRegion/SAM2 region.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output_path", default="scratch/textregion_regions_grid.png")
    parser.add_argument("--textregion_repo", default="scratch/TextRegion")
    parser.add_argument("--textregion_sam2_checkpoint", default="scratch/TextRegion/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--textregion_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--textregion_points_per_side", type=int, default=16)
    parser.add_argument("--max_regions", type=int, default=32)
    parser.add_argument("--min_mask_pixels", type=int, default=8)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.65)
    return parser


if __name__ == "__main__":
    render(build_parser().parse_args())
