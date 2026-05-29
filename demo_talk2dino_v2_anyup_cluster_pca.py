import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from batch_extract_features import adaptive_spherical_kmeans
from demo_talk2dino_v2_single_image import (
    DEFAULT_ANYUP_MODEL,
    DEFAULT_IMAGE_PATH,
    DEFAULT_MODEL_ID,
    build_hr_image_tensor,
    extract_lr_feature_map,
    load_talk2dino_model,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Cluster and visualize AnyUp-upsampled Talk2DINO v2 features on a single image."
    )
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="HF model id.")
    parser.add_argument("--output_path", default=None, help="Output figure path.")
    parser.add_argument("--num_clusters", type=int, default=8, help="Maximum number of clusters.")
    parser.add_argument("--min_cluster_pixels", type=int, default=8, help="Minimum cluster size before reassignment.")
    parser.add_argument("--merge_similarity", type=float, default=0.95, help="Merge clusters above this cosine similarity.")
    parser.add_argument("--representative_mode", choices=["mean", "geometric_median"], default="mean")
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL, help="torch.hub AnyUp entrypoint.")
    parser.add_argument("--anyup_use_natten", action="store_true", help="Use NATTEN-based AnyUp model.")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=None, help="Optional AnyUp q_chunk_size.")
    return parser


def build_palette(num_clusters: int):
    base = np.array(
        [
            [255, 99, 132],
            [54, 162, 235],
            [255, 206, 86],
            [75, 192, 192],
            [153, 102, 255],
            [255, 159, 64],
            [199, 199, 199],
            [83, 102, 255],
            [40, 180, 99],
            [214, 48, 49],
        ],
        dtype=np.uint8,
    )
    if num_clusters <= len(base):
        return base[:num_clusters]
    extra = np.random.default_rng(0).integers(0, 255, size=(num_clusters - len(base), 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)


def upsample_anyup_features(model, upsampler, image: Image.Image, device: str, q_chunk_size=None, output_size=None):
    lr_features = extract_lr_feature_map(model, image, device)
    hr_image = build_hr_image_tensor(image, device)
    kwargs = {}
    if q_chunk_size is not None:
        kwargs["q_chunk_size"] = q_chunk_size
    if output_size is not None:
        kwargs["output_size"] = output_size
    with torch.no_grad():
        hr_features = upsampler(hr_image, lr_features, **kwargs)
    hr_features = hr_features.to(device)
    hr_features = hr_features / hr_features.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return hr_features


def compute_pca_rgb(feature_map: torch.Tensor):
    b, c, h, w = feature_map.shape
    assert b == 1
    flat = feature_map[0].permute(1, 2, 0).reshape(-1, c)
    flat = flat - flat.mean(dim=0, keepdim=True)
    u, s, v = torch.pca_lowrank(flat, q=3)
    proj = flat @ v[:, :3]
    proj = proj.reshape(h, w, 3)
    proj = proj - proj.amin(dim=(0, 1), keepdim=True)
    proj = proj / proj.amax(dim=(0, 1), keepdim=True).clamp_min(1e-8)
    return proj.detach().cpu().numpy()


def cluster_feature_map(
    feature_map: torch.Tensor,
    num_clusters: int,
    min_cluster_pixels: int,
    merge_similarity: float,
    representative_mode: str = "mean",
):
    b, c, h, w = feature_map.shape
    assert b == 1
    flat = feature_map[0].permute(1, 2, 0).reshape(-1, c)
    centers, labels = adaptive_spherical_kmeans(
        flat,
        max_clusters=num_clusters,
        min_cluster_pixels=min_cluster_pixels,
        merge_similarity=merge_similarity,
        representative_mode=representative_mode,
    )
    label_map = labels.reshape(h, w)
    return centers, label_map


def render_demo(
    image_path: Path,
    model_id: str,
    output_path: Path,
    num_clusters: int,
    min_cluster_pixels: int,
    merge_similarity: float,
    anyup_entrypoint: str,
    anyup_use_natten: bool,
    anyup_q_chunk_size,
    representative_mode: str,
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
    image_np = np.array(image)

    hr_features = upsample_anyup_features(model, upsampler, image, device, q_chunk_size=anyup_q_chunk_size)
    pca_rgb = compute_pca_rgb(hr_features)
    centers, label_map = cluster_feature_map(
        hr_features,
        num_clusters=num_clusters,
        min_cluster_pixels=min_cluster_pixels,
        merge_similarity=merge_similarity,
        representative_mode=representative_mode,
    )

    palette = build_palette(int(centers.shape[0]))
    cluster_rgb = palette[label_map.detach().cpu().numpy()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(image_np)
    axes[1].imshow(pca_rgb, alpha=0.6, interpolation="bilinear")
    axes[1].set_title("AnyUp PCA overlay")
    axes[1].axis("off")

    axes[2].imshow(image_np)
    axes[2].imshow(cluster_rgb, alpha=0.55, interpolation="nearest")
    axes[2].set_title(
        f"AnyUp cluster overlay\nrequested={num_clusters}, final={centers.shape[0]}\n"
        f"min_pixels={min_cluster_pixels}, merge_sim={merge_similarity}\n"
        f"representative={representative_mode}"
    )
    axes[2].axis("off")

    fig.suptitle(f"Talk2DINO v2 + AnyUp feature analysis\n{image_path.name}", fontsize=14)
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
        output_path = Path("scratch") / f"{image_path.stem}_talk2dino_v2_anyup_cluster_pca.png"

    render_demo(
        image_path=image_path,
        model_id=args.model_id,
        output_path=output_path,
        num_clusters=args.num_clusters,
        min_cluster_pixels=args.min_cluster_pixels,
        merge_similarity=args.merge_similarity,
        anyup_entrypoint=args.anyup_entrypoint,
        anyup_use_natten=args.anyup_use_natten,
        anyup_q_chunk_size=args.anyup_q_chunk_size,
        representative_mode=args.representative_mode,
    )


if __name__ == "__main__":
    main()
