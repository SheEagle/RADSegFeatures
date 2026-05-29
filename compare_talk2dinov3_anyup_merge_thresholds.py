import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Compare Talk2DINOv3+AnyUp cluster overlays across merge thresholds.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_path", default="scratch/talk2dinov3_anyup_merge_threshold_compare.png")
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--merge_values", nargs="+", type=float, default=[0.90, 0.93, 0.95, 0.97])
    parser.add_argument("--min_cluster_pixels", type=int, default=16)
    parser.add_argument("--representative_mode", choices=["mean", "geometric_median"], default="mean")
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL)
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=10)
    parser.add_argument("--anyup_output_size", nargs=2, type=int, default=[384, 384], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--panel_width", type=int, default=640)
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
    for merge in args.merge_values:
        print(f"Clustering merge={merge}...")
        centers, label_map = cluster_feature_map(
            hr_features,
            num_clusters=args.num_clusters,
            min_cluster_pixels=args.min_cluster_pixels,
            merge_similarity=merge,
            representative_mode=args.representative_mode,
        )
        palette = build_palette(int(centers.shape[0]))
        cluster_rgb = palette[label_map.detach().cpu().numpy()]
        panels.append(
            (
                f"K={args.num_clusters}, merge={merge}\nfinal={centers.shape[0]}, min_pixels={args.min_cluster_pixels}",
                overlay_to_image(image, cluster_rgb),
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

    fig.suptitle(f"Talk2DINOv3 + AnyUp merge-threshold comparison\n{image_path.name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to: {output_path}")


def main():
    render_comparison(build_parser().parse_args())


if __name__ == "__main__":
    main()
