import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from PIL import Image

from demo_talk2dino_v2_anyup_cluster_pca import (
    DEFAULT_ANYUP_MODEL,
    DEFAULT_IMAGE_PATH,
    load_talk2dino_model,
    upsample_anyup_features,
)
from demo_talk2dino_v2_anyup_cluster_search import (
    encode_prompts,
    make_cluster_legend_overlay,
    make_similarity_overlay,
    normalize_negative_prompts,
    sam3_feature_map,
    score_clusters,
)


def build_sam3_cluster_bank(hr_features, image, args, device):
    prompts = args.sam3_prompts or [args.sam3_prompt]
    combined_label_map = None
    combined_centers = []

    for prompt in prompts:
        print(f"Adding SAM3 clusters from prompt='{prompt}'...")
        centers, label_map = sam3_feature_map(
            hr_features,
            image,
            query=prompt,
            repo_path=args.sam3_repo,
            checkpoint_path=args.sam3_checkpoint_path,
            resolution=args.sam3_resolution,
            confidence_threshold=args.sam3_confidence_threshold,
            max_masks=args.sam3_max_masks,
            min_mask_pixels=args.min_cluster_pixels,
            device=device,
        )
        if combined_label_map is None:
            combined_label_map = torch.full_like(label_map, -1)

        for old_cluster_id in range(centers.shape[0]):
            mask = label_map == old_cluster_id
            assign_mask = (combined_label_map < 0) & mask
            if not assign_mask.any():
                continue
            new_cluster_id = len(combined_centers)
            combined_centers.append(centers[old_cluster_id])
            combined_label_map[assign_mask] = new_cluster_id

    if not combined_centers:
        h, w = hr_features.shape[-2:]
        flat = hr_features[0].permute(1, 2, 0).reshape(-1, hr_features.shape[1])
        combined_label_map = torch.zeros((h, w), device=hr_features.device, dtype=torch.long)
        return F.normalize(flat.mean(dim=0, keepdim=True), dim=-1), combined_label_map

    return torch.stack(combined_centers, dim=0), combined_label_map


def render_query(
    image,
    label_map,
    centers,
    model,
    query,
    negative_text,
    temperature,
    top_clusters,
    output_path,
    device,
    sam3_prompt,
):
    negative_prompts = normalize_negative_prompts(negative_text)
    text_vectors = encode_prompts(model, [query] + negative_prompts, device)
    positive_vector = text_vectors[0]
    negative_vectors = [vec for vec in text_vectors[1:]]

    pos_scores, scores = score_clusters(centers, positive_vector, negative_vectors, temperature)
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

    cluster_id_map = label_map.detach().cpu().numpy()
    overlay, heatmap = make_similarity_overlay(image, cluster_id_map, cluster_items)
    cluster_overlay, legend_entries = make_cluster_legend_overlay(
        image,
        cluster_id_map,
        cluster_items,
        top_clusters=top_clusters,
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
    axes[1].set_title(f"Pure cluster search\nquery='{query}' vs [{', '.join(negative_prompts)}]")
    axes[1].axis("off")

    axes[2].imshow(cluster_overlay)
    cluster_text = ", ".join(
        f"{item['cluster_id']}:{item['score']:.4f}" for item in cluster_items[:top_clusters]
    )
    axes[2].set_title(
        f"SAM3 prompt='{sam3_prompt}' clusters\n"
        f"clusters={cluster_text}\n"
        f"regions={centers.shape[0]}"
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {query}: {output_path}")


def render(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        output_size=tuple(args.anyup_output_size) if args.anyup_output_size is not None else None,
    )

    prompt_label = ", ".join(args.sam3_prompts) if args.sam3_prompts else args.sam3_prompt
    print(f"Building SAM3 cluster bank once with prompts={prompt_label!r}...")
    centers, label_map = build_sam3_cluster_bank(hr_features, image, args, device)
    print(f"SAM3 cluster count: {centers.shape[0]}")

    stem = image_path.stem
    for query in args.queries:
        output_path = output_dir / f"{stem}_sam3_pure_search_{query}.png"
        render_query(
            image=image,
            label_map=label_map,
            centers=centers,
            model=model,
            query=query,
            negative_text=args.negative_text,
            temperature=args.temperature,
            top_clusters=args.top_clusters,
            output_path=output_path,
            device=device,
            sam3_prompt=prompt_label,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Search fixed SAM3 clusters without query-time SAM3 refinement.")
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--model_id", default="lorebianchi98/Talk2DINOv3-ViTL")
    parser.add_argument("--queries", nargs="+", default=["tree", "building", "car"])
    parser.add_argument("--negative_text", default="background")
    parser.add_argument("--output_dir", default="scratch")
    parser.add_argument("--sam3_prompt", default="object")
    parser.add_argument("--sam3_prompts", nargs="+", default=None)
    parser.add_argument("--sam3_repo", default="scratch/sam3")
    parser.add_argument("--sam3_checkpoint_path", default="sam3.pt")
    parser.add_argument("--sam3_resolution", type=int, default=1008)
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.25)
    parser.add_argument("--sam3_max_masks", type=int, default=64)
    parser.add_argument("--min_cluster_pixels", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--top_clusters", type=int, default=6)
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL)
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=10)
    parser.add_argument("--anyup_output_size", nargs=2, type=int, default=[384, 384], metavar=("WIDTH", "HEIGHT"))
    return parser


if __name__ == "__main__":
    render(build_parser().parse_args())
