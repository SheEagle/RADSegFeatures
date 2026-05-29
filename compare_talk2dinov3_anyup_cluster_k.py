import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from demo_talk2dino_v2_anyup_cluster_pca import (
    build_palette,
    cluster_feature_map,
    compute_pca_rgb,
    upsample_anyup_features,
)
from demo_talk2dino_v2_single_image import DEFAULT_ANYUP_MODEL, load_talk2dino_model


DEFAULT_IMAGE_PATH = r"D:\RADSeg\1a2b81a5-845f-5cb1-b4b2-0e0e3df27bf2.jpeg"
DEFAULT_MODEL_ID = "lorebianchi98/Talk2DINOv3-ViTL"


def build_parser():
    parser = argparse.ArgumentParser(description="Compare Talk2DINOv3+AnyUp cluster overlays across K values.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--min_cluster_pixels", type=int, default=16)
    parser.add_argument("--merge_similarity", type=float, default=0.97)
    parser.add_argument("--representative_mode", choices=["mean", "geometric_median"], default="mean")
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL)
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=10)
    parser.add_argument("--anyup_output_size", nargs=2, type=int, default=[384, 384], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--panel_width", type=int, default=560)
    return parser


def resize_into_canvas(image, canvas_size, background=(255, 255, 255)):
    canvas_w, canvas_h = canvas_size
    image = image.convert("RGB")
    scale = min(canvas_w / image.width, canvas_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_size, background)
    offset = ((canvas_w - new_size[0]) // 2, (canvas_h - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def overlay_to_image(image, overlay_rgb, alpha=0.56):
    overlay = torch.from_numpy(overlay_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    overlay = F.interpolate(overlay, size=(image.height, image.width), mode="nearest")[0].permute(1, 2, 0).numpy()
    base = np.asarray(image).astype(np.float32) / 255.0
    mixed = base * (1.0 - alpha) + overlay * alpha
    return Image.fromarray(np.clip(mixed * 255.0, 0, 255).astype(np.uint8))


def pca_to_image(image, pca_rgb, alpha=0.62):
    overlay = torch.from_numpy(pca_rgb).permute(2, 0, 1).unsqueeze(0).float()
    overlay = F.interpolate(overlay, size=(image.height, image.width), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    base = np.asarray(image).astype(np.float32) / 255.0
    mixed = base * (1.0 - alpha) + overlay * alpha
    return Image.fromarray(np.clip(mixed * 255.0, 0, 255).astype(np.uint8))


def draw_panel(ax, panel_image, title, canvas_size):
    ax.imshow(resize_into_canvas(panel_image, canvas_size))
    ax.set_title(title, fontsize=12, pad=8)
    ax.axis("off")


def render_comparison(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_path = Path(args.output_path) if args.output_path else Path("scratch") / (
        f"{image_path.stem}_talk2dinov3_anyup_k_compare.png"
    )
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
    aspect = image.height / image.width
    canvas_size = (args.panel_width, max(1, int(args.panel_width * aspect)))

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

    panels = [("Original", image), (f"AnyUp PCA\n{hr_features.shape[-2]}x{hr_features.shape[-1]} feature map", pca_to_image(image, pca_rgb))]

    for k in args.k_values:
        print(f"Clustering K={k}...")
        centers, label_map = cluster_feature_map(
            hr_features,
            num_clusters=k,
            min_cluster_pixels=args.min_cluster_pixels,
            merge_similarity=args.merge_similarity,
            representative_mode=args.representative_mode,
        )
        palette = build_palette(int(centers.shape[0]))
        cluster_rgb = palette[label_map.detach().cpu().numpy()]
        panels.append((
            f"K={k}, final={centers.shape[0]}\n"
            f"min_pixels={args.min_cluster_pixels}, merge={args.merge_similarity}\n"
            f"rep={args.representative_mode}",
            overlay_to_image(image, cluster_rgb),
        ))

    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig_w = 4.8 * cols
    fig_h = fig_w * (canvas_size[1] / canvas_size[0]) * rows / cols + 0.8 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, panel_image) in zip(axes.flat, panels):
        draw_panel(ax, panel_image, title, canvas_size)

    fig.suptitle(f"Talk2DINOv3 + AnyUp cluster/PCA comparison\n{image_path.name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to: {output_path}")


def main():
    args = build_parser().parse_args()
    render_comparison(args)


if __name__ == "__main__":
    main()
