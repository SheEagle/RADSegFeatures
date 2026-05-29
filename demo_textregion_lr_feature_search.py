import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import torch

from demo_talk2dino_v2_single_image import (
    DEFAULT_IMAGE_PATH,
    load_talk2dino_model,
    extract_lr_feature_map,
)
from demo_talk2dino_v2_anyup_cluster_search import (
    encode_prompts,
    feature_regions_from_masks,
    load_textregion_mask_generator,
    make_cluster_legend_overlay,
    make_similarity_overlay,
    normalize_negative_prompts,
    score_clusters,
)


@torch.no_grad()
def extract_final_value_feature_map(model, image, device):
    if not hasattr(model, "model") or not hasattr(model.model, "blocks"):
        raise ValueError("Value feature extraction expects a Talk2DINO model with a DINO/EVA backbone.")

    captured = {}

    def hook(_module, _inputs, output):
        captured["qkv"] = output

    attn = model.model.blocks[-1].attn
    handle = attn.qkv.register_forward_hook(hook)
    try:
        image_tensor = model.image_transforms(image).to(device).unsqueeze(0)
        model.model.forward_features(image_tensor)
    finally:
        handle.remove()

    if "qkv" not in captured:
        raise RuntimeError("Failed to capture final-block qkv output.")

    qkv_out = captured["qkv"]
    batch_size, num_tokens, three_channels = qkv_out.shape
    channels = three_channels // 3
    num_heads = getattr(attn, "num_heads", None) or getattr(model, "num_attn_heads", 16)
    values = qkv_out.reshape(batch_size, num_tokens, 3, num_heads, channels // num_heads)
    values = values[:, :, 2].permute(0, 2, 1, 3).reshape(batch_size, num_tokens, channels)

    num_global_tokens = getattr(model, "num_global_tokens", 5)
    patch_values = values[:, num_global_tokens:]
    side = int(patch_values.shape[1] ** 0.5)
    if side * side != patch_values.shape[1]:
        raise ValueError(f"Value token count {patch_values.shape[1]} is not a square grid.")

    feature_map = patch_values.reshape(batch_size, side, side, channels).permute(0, 3, 1, 2)
    return torch.nn.functional.normalize(feature_map, dim=1)


@torch.no_grad()
def textregion_lr_feature_map(
    feature_map,
    image,
    repo_path,
    model_cfg,
    checkpoint,
    points_per_side,
    max_masks,
    min_mask_pixels,
    device,
):
    generator = load_textregion_mask_generator(
        repo_path=repo_path,
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        points_per_side=points_per_side,
        device=device,
    )
    import numpy as np

    image_np = np.array(image.convert("RGB"), copy=True)
    image_tensor = torch.from_numpy(image_np).to(device=device, dtype=torch.float32)
    image_tensor_for_sam2 = torch.stack([image_tensor])
    image_tensor_for_sam2 = generator.predictor._transforms(image_tensor_for_sam2)
    ori_shape = image_np.shape[:2]
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device == "cuda"):
        sam2_masks = generator.generate_for_batch(image_tensor_for_sam2, [ori_shape], None)

    masks = torch.stack([mask["segmentations"] for mask in sam2_masks[0]])
    return feature_regions_from_masks(
        feature_map,
        masks,
        min_mask_pixels=min_mask_pixels,
        max_masks=max_masks,
    )


def render(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = Path(args.image_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model: {args.model_id}")
    model = load_talk2dino_model(args.model_id, device)

    image = Image.open(image_path).convert("RGB")
    if args.feature_source == "value":
        lr_features = extract_final_value_feature_map(model, image, device)
    else:
        lr_features = extract_lr_feature_map(model, image, device)
    print(f"{args.feature_source} feature map: {tuple(lr_features.shape)}")

    print("Segmenting image with TextRegion/SAM2 masks...")
    centers, label_map = textregion_lr_feature_map(
        lr_features,
        image,
        repo_path=args.textregion_repo,
        model_cfg=args.textregion_model_cfg,
        checkpoint=args.textregion_sam2_checkpoint,
        points_per_side=args.textregion_points_per_side,
        max_masks=args.textregion_max_masks,
        min_mask_pixels=args.min_cluster_pixels,
        device=device,
    )
    print(f"Segmentation produced {centers.shape[0]} regions.")

    negative_prompts = normalize_negative_prompts(args.negative_text)
    text_vectors = encode_prompts(model, [args.query] + negative_prompts, device)
    positive_vector = text_vectors[0]
    negative_vectors = [vec for vec in text_vectors[1:]]

    pos_scores, scores = score_clusters(centers, positive_vector, negative_vectors, args.temperature)
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

    label_map_np = label_map.detach().cpu().numpy()
    overlay, heatmap = make_similarity_overlay(image, label_map_np, cluster_items)
    cluster_overlay, legend_entries = make_cluster_legend_overlay(
        image,
        label_map_np,
        cluster_items,
        top_clusters=args.top_clusters,
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
    axes[1].set_title(f"LR patch region score map\nquery='{args.query}' vs [{', '.join(negative_prompts)}]")
    axes[1].axis("off")

    axes[2].imshow(cluster_overlay)
    cluster_text = ", ".join(
        f"{item['cluster_id']}:{item['score']:.4f}" for item in cluster_items[: args.top_clusters]
    )
    axes[2].set_title(
        f"TextRegion + LR patch pooling\n"
        f"clusters={cluster_text}\n"
        f"{args.feature_source}={lr_features.shape[-2]}x{lr_features.shape[-1]}, regions={centers.shape[0]}"
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
        axes[3].legend(handles=handles, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="TextRegion/SAM2 search with Talk2DINOv3 low-res patch feature pooling.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model_id", default="lorebianchi98/Talk2DINOv3-ViTL")
    parser.add_argument("--query", default="river")
    parser.add_argument("--negative_text", default="background")
    parser.add_argument("--output_path", default="scratch/textregion_lr_feature_search_river.png")
    parser.add_argument("--feature_source", choices=["patch", "value"], default="patch")
    parser.add_argument("--textregion_repo", default="scratch/TextRegion")
    parser.add_argument("--textregion_sam2_checkpoint", default="scratch/TextRegion/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--textregion_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--textregion_points_per_side", type=int, default=16)
    parser.add_argument("--textregion_max_masks", type=int, default=32)
    parser.add_argument("--min_cluster_pixels", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--top_clusters", type=int, default=6)
    return parser


if __name__ == "__main__":
    render(build_parser().parse_args())
