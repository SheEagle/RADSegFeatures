import argparse
import importlib
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from PIL import Image

from batch_extract_features import adaptive_spherical_kmeans
from demo_talk2dino_v2_anyup_cluster_pca import (
    DEFAULT_ANYUP_MODEL,
    DEFAULT_IMAGE_PATH,
    DEFAULT_MODEL_ID,
    load_talk2dino_model,
    upsample_anyup_features,
)


def geometric_median(points, max_iters=30, tol=1e-5):
    if points.shape[0] == 0:
        return points[:0]
    if points.shape[0] == 1:
        return F.normalize(points[0], dim=0)

    estimate = points.mean(dim=0)
    for _ in range(max_iters):
        distances = torch.linalg.norm(points - estimate.unsqueeze(0), dim=1).clamp_min(1e-8)
        weights = 1.0 / distances
        next_estimate = (points * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        if torch.linalg.norm(next_estimate - estimate) < tol:
            estimate = next_estimate
            break
        estimate = next_estimate
    return F.normalize(estimate, dim=0)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Single-image Talk2DINO v2 + AnyUp cluster search with RADSeg-style heatmap visualization."
    )
    parser.add_argument("--image_path", default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="HF model id.")
    parser.add_argument("--query", default="river", help="Positive query text.")
    parser.add_argument("--negative_text", default="background", help="Comma-separated negative prompts.")
    parser.add_argument("--output_path", default=None, help="Output figure path.")
    parser.add_argument(
        "--segmentation_mode",
        choices=["epoc", "textregion", "sam3", "kmeans"],
        default="epoc",
        help="Region source: EPOC subobjects, TextRegion/SAM2 masks, SAM3 masks, or spherical K-Means.",
    )
    parser.add_argument("--num_clusters", type=int, default=10, help="Maximum number of clusters.")
    parser.add_argument("--min_cluster_pixels", type=int, default=8, help="Minimum cluster size before reassignment.")
    parser.add_argument("--merge_similarity", type=float, default=0.95, help="Merge clusters above this cosine similarity.")
    parser.add_argument(
        "--subobjects_repo",
        default=None,
        help="Path to a local clone of https://github.com/ChenDelong1999/subobjects.",
    )
    parser.add_argument(
        "--subobject_checkpoint",
        default="chendelong/DirectSAM-b0-1024px-sa1b-2ep-1017",
        help="DirectSAM checkpoint for EPOC subobject segmentation.",
    )
    parser.add_argument("--subobject_resolution", type=int, default=1024, help="Square resolution used by EPOC.")
    parser.add_argument("--subobject_threshold", type=float, default=0.1, help="EPOC boundary threshold.")
    parser.add_argument("--subobject_max_tokens", type=int, default=64, help="Maximum EPOC masks to score.")
    parser.add_argument("--subobject_crop", type=int, default=1, help="EPOC crop grid size.")
    parser.add_argument("--textregion_repo", default="scratch/TextRegion", help="Path to a local TextRegion clone.")
    parser.add_argument(
        "--textregion_sam2_checkpoint",
        default="scratch/TextRegion/checkpoints/sam2.1_hiera_large.pt",
        help="Path to the SAM2.1 checkpoint used by TextRegion.",
    )
    parser.add_argument(
        "--textregion_model_cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config name relative to the TextRegion sam2 package.",
    )
    parser.add_argument("--textregion_points_per_side", type=int, default=16, help="SAM2 grid density.")
    parser.add_argument("--textregion_max_masks", type=int, default=64, help="Maximum TextRegion/SAM2 masks to score.")
    parser.add_argument("--textregion_pred_iou_thresh", type=float, default=0.6, help="SAM2 mask quality threshold.")
    parser.add_argument("--textregion_stability_score_thresh", type=float, default=0.6, help="SAM2 mask stability threshold.")
    parser.add_argument("--textregion_box_nms_thresh", type=float, default=0.9, help="SAM2 box NMS threshold.")
    parser.add_argument(
        "--region_representative_mode",
        choices=["mean", "geometric_median"],
        default="mean",
        help="How to pool dense features inside TextRegion/SAM3 masks.",
    )
    parser.add_argument(
        "--mask_pooling_mode",
        choices=["hard", "soft"],
        default="hard",
        help="Use binary masks or raw soft mask weights when pooling region features.",
    )
    parser.add_argument("--sam3_repo", default="scratch/sam3", help="Path to a local facebookresearch/sam3 clone.")
    parser.add_argument("--sam3_checkpoint_path", default="sam3.pt", help="Path to a local SAM3 checkpoint.")
    parser.add_argument("--sam3_resolution", type=int, default=1008, help="Square resolution used by SAM3.")
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.35, help="SAM3 text-prompt mask threshold.")
    parser.add_argument("--sam3_max_masks", type=int, default=64, help="Maximum SAM3 masks to score.")
    parser.add_argument("--temperature", type=float, default=10.0, help="Contrastive softmax temperature.")
    parser.add_argument("--top_clusters", type=int, default=6, help="How many cluster hits to print in the title.")
    parser.add_argument("--anyup_entrypoint", default=DEFAULT_ANYUP_MODEL, help="torch.hub AnyUp entrypoint.")
    parser.add_argument("--anyup_use_natten", action="store_true", help="Use NATTEN-based AnyUp model.")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=None, help="Optional AnyUp q_chunk_size.")
    parser.add_argument(
        "--anyup_output_size",
        nargs=2,
        type=int,
        default=[384, 384],
        metavar=("WIDTH", "HEIGHT"),
        help="AnyUp output size, default 384 384 for practical speed.",
    )
    return parser


def normalize_negative_prompts(negative_text):
    if not negative_text:
        return ["background"]
    prompts = [part.strip() for part in str(negative_text).split(",") if part.strip()]
    return prompts or ["background"]


@torch.no_grad()
def encode_prompts(model, prompts, device):
    embeddings = model.encode_text(prompts)
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    embeddings = embeddings.to(device)
    return F.normalize(embeddings, dim=-1)


def cluster_feature_map(feature_map: torch.Tensor, num_clusters: int, min_cluster_pixels: int, merge_similarity: float):
    b, c, h, w = feature_map.shape
    assert b == 1
    flat = feature_map[0].permute(1, 2, 0).reshape(-1, c)
    centers, labels = adaptive_spherical_kmeans(
        flat,
        max_clusters=num_clusters,
        min_cluster_pixels=min_cluster_pixels,
        merge_similarity=merge_similarity,
    )
    label_map = labels.reshape(h, w)
    return centers, label_map


def load_epoc_segmenter(
    repo_path,
    checkpoint,
    image_resolution,
    threshold,
    max_tokens,
    crop,
    device,
):
    candidate_paths = [
        repo_path,
        os.environ.get("SUBOBJECTS_REPO"),
        Path.cwd() / "subobjects",
        Path.cwd() / "scratch" / "subobjects_repo",
    ]
    for path in candidate_paths:
        if path and Path(path).exists():
            resolved = str(Path(path).resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

    try:
        module = importlib.import_module("subobjects.token_segmentation.epoc")
        segmenter_cls = getattr(module, "EPOCSegmenter")
    except Exception as exc:
        raise ImportError(
            "EPOC mode requires ChenDelong1999/subobjects. Clone it and pass "
            "--subobjects_repo PATH, or set SUBOBJECTS_REPO."
        ) from exc

    segmenter = segmenter_cls(
        image_resolution=image_resolution,
        checkpoint=checkpoint,
        threshold=threshold,
        max_tokens=max_tokens,
        crop=crop,
        device=device,
    )
    if device == "cpu":
        segmenter.model = segmenter.model.float()
    return segmenter


def subobject_feature_map(
    feature_map: torch.Tensor,
    image: Image.Image,
    segmenter,
    min_mask_pixels: int,
):
    b, c, h, w = feature_map.shape
    assert b == 1

    flat = feature_map[0].permute(1, 2, 0).reshape(-1, c)
    flat = F.normalize(flat.float(), dim=-1)

    masks = segmenter(image)
    if not isinstance(masks, torch.Tensor):
        masks = torch.as_tensor(masks)
    masks = masks[0].to(feature_map.device).bool()
    areas = masks.flatten(1).sum(dim=1)
    masks = masks[areas > 0]
    if masks.numel() == 0:
        label_map = torch.zeros((h, w), device=feature_map.device, dtype=torch.long)
        return flat.mean(dim=0, keepdim=True), label_map

    resized_masks = F.interpolate(
        masks[:, None].float(),
        size=(h, w),
        mode="nearest",
    )[:, 0].bool()
    resized_areas = resized_masks.flatten(1).sum(dim=1)
    resized_masks = resized_masks[resized_areas >= max(1, min_mask_pixels)]
    if resized_masks.numel() == 0:
        label_map = torch.zeros((h, w), device=feature_map.device, dtype=torch.long)
        return flat.mean(dim=0, keepdim=True), label_map

    label_map = torch.full((h, w), -1, device=feature_map.device, dtype=torch.long)
    centers = []
    for mask in resized_masks:
        cluster_id = len(centers)
        token_mask = mask.reshape(-1)
        if not token_mask.any():
            continue
        center = flat[token_mask].mean(dim=0)
        center = F.normalize(center, dim=0)
        centers.append(center)
        label_map[(label_map < 0) & mask] = cluster_id

    if not centers:
        label_map.fill_(0)
        return flat.mean(dim=0, keepdim=True), label_map

    centers = torch.stack(centers, dim=0)
    flat_labels = label_map.reshape(-1)
    unassigned = flat_labels < 0
    if unassigned.any():
        sims = torch.matmul(flat[unassigned], centers.transpose(0, 1))
        flat_labels[unassigned] = torch.argmax(sims, dim=-1)

    return centers, label_map


def feature_regions_from_masks(
    feature_map: torch.Tensor,
    masks: torch.Tensor,
    min_mask_pixels: int,
    max_masks=None,
    assign_uncovered=True,
    representative_mode="mean",
    mask_pooling_mode="hard",
):
    b, c, h, w = feature_map.shape
    assert b == 1

    flat = feature_map[0].permute(1, 2, 0).reshape(-1, c)
    flat = F.normalize(flat.float(), dim=-1)

    if not isinstance(masks, torch.Tensor):
        masks = torch.as_tensor(masks)
    masks = masks.to(feature_map.device)
    if masks.numel() == 0:
        label_map = torch.zeros((h, w), device=feature_map.device, dtype=torch.long)
        return F.normalize(flat.mean(dim=0, keepdim=True), dim=-1), label_map

    if mask_pooling_mode == "soft":
        if representative_mode != "mean":
            raise ValueError("soft mask pooling currently supports representative_mode='mean' only.")
        masks = masks.float().clamp_min(0)
        max_vals = masks.flatten(1).amax(dim=1).clamp_min(1e-8)
        masks = masks / max_vals[:, None, None]
        support_masks = masks > 1e-6
    elif mask_pooling_mode == "hard":
        masks = masks.bool()
        support_masks = masks
    else:
        raise ValueError(f"Unsupported mask_pooling_mode: {mask_pooling_mode}")

    areas = support_masks.flatten(1).sum(dim=1)
    keep = areas > 0
    masks = masks[keep]
    support_masks = support_masks[keep]
    areas = areas[keep]
    if masks.numel() == 0:
        label_map = torch.zeros((h, w), device=feature_map.device, dtype=torch.long)
        return F.normalize(flat.mean(dim=0, keepdim=True), dim=-1), label_map

    order = torch.argsort(areas, descending=True)
    if max_masks is not None and max_masks > 0:
        order = order[:max_masks]
    masks = masks[order]
    support_masks = support_masks[order]

    if mask_pooling_mode == "soft":
        resized_weights = F.interpolate(
            masks[:, None].float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[:, 0].clamp_min(0)
        resized_masks = resized_weights > 1e-6
    else:
        resized_weights = None
        resized_masks = F.interpolate(
            support_masks[:, None].float(),
            size=(h, w),
            mode="nearest",
        )[:, 0].bool()

    resized_areas = resized_masks.flatten(1).sum(dim=1)
    keep = resized_areas >= max(1, min_mask_pixels)
    resized_masks = resized_masks[keep]
    if resized_weights is not None:
        resized_weights = resized_weights[keep]
    if resized_masks.numel() == 0:
        label_map = torch.zeros((h, w), device=feature_map.device, dtype=torch.long)
        return F.normalize(flat.mean(dim=0, keepdim=True), dim=-1), label_map

    label_map = torch.full((h, w), -1, device=feature_map.device, dtype=torch.long)
    centers = []
    for idx, mask in enumerate(resized_masks):
        token_mask = mask.reshape(-1)
        assign_mask = (label_map < 0) & mask
        if not token_mask.any() or not assign_mask.any():
            continue
        if mask_pooling_mode == "soft":
            token_weights = resized_weights[idx].reshape(-1)
            weight_sum = token_weights.sum()
            if weight_sum <= 0:
                continue
            center = F.normalize((flat * token_weights[:, None]).sum(dim=0) / weight_sum, dim=0)
        elif representative_mode == "mean":
            center = F.normalize(flat[token_mask].mean(dim=0), dim=0)
        elif representative_mode == "geometric_median":
            center = geometric_median(flat[token_mask])
        else:
            raise ValueError(f"Unsupported representative_mode: {representative_mode}")
        centers.append(center)
        label_map[assign_mask] = len(centers) - 1

    if not centers:
        label_map.fill_(0)
        return F.normalize(flat.mean(dim=0, keepdim=True), dim=-1), label_map

    centers = torch.stack(centers, dim=0)
    flat_labels = label_map.reshape(-1)
    unassigned = flat_labels < 0
    if assign_uncovered and unassigned.any():
        sims = torch.matmul(flat[unassigned], centers.transpose(0, 1))
        flat_labels[unassigned] = torch.argmax(sims, dim=-1)

    return centers, label_map


def load_textregion_mask_generator(
    repo_path,
    model_cfg,
    checkpoint,
    points_per_side,
    pred_iou_thresh,
    stability_score_thresh,
    box_nms_thresh,
    device,
):
    repo_root = Path(repo_path).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"TextRegion repo not found: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from sam2.build_sam import build_sam2
    from sam2.custom_automatic_mask_generator import CustomAutomaticMaskGenerator

    model = build_sam2(
        model_cfg,
        str(Path(checkpoint).resolve()),
        device=device,
        apply_postprocessing=False,
    )
    return CustomAutomaticMaskGenerator(
        prompt_method="grid",
        model=model,
        point_grids=None,
        min_mask_region_area=0,
        points_per_side=points_per_side,
        points_per_batch=2048,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        multimask_output=True,
        fuse_mask=True,
        fuse_mask_threshold=0.8,
    )


@torch.no_grad()
def textregion_feature_map(
    feature_map: torch.Tensor,
    image: Image.Image,
    repo_path,
    model_cfg,
    checkpoint,
    points_per_side,
    pred_iou_thresh,
    stability_score_thresh,
    box_nms_thresh,
    max_masks,
    min_mask_pixels,
    representative_mode,
    mask_pooling_mode,
    device,
):
    generator = load_textregion_mask_generator(
        repo_path=repo_path,
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        device=device,
    )
    image_np = np.asarray(image.convert("RGB"))
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
        representative_mode=representative_mode,
        mask_pooling_mode=mask_pooling_mode,
    )


@torch.no_grad()
def sam3_feature_map(
    feature_map: torch.Tensor,
    image: Image.Image,
    query: str,
    repo_path,
    checkpoint_path,
    resolution: int,
    confidence_threshold: float,
    max_masks: int,
    min_mask_pixels: int,
    device,
    representative_mode="mean",
    mask_pooling_mode="hard",
):
    repo_path = Path(repo_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"SAM3 repo not found: {repo_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    sam3_model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
    )
    sam3_model = sam3_model.to(device).eval()
    processor = Sam3Processor(
        sam3_model,
        resolution=resolution,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device == "cuda"):
        state = processor.set_image(image)
        outputs = processor.set_text_prompt(state=state, prompt=query)

    raw_masks = outputs.get("masks")
    if raw_masks is None or raw_masks.numel() == 0:
        h, w = feature_map.shape[-2:]
        label_map = torch.zeros((h, w), device=feature_map.device, dtype=torch.long)
        flat = feature_map[0].permute(1, 2, 0).reshape(-1, feature_map.shape[1])
        return F.normalize(flat.mean(dim=0, keepdim=True), dim=-1), label_map

    masks = raw_masks.detach().cpu()
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.dim() == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    masks = masks.bool()

    del processor
    del sam3_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return feature_regions_from_masks(
        feature_map,
        masks,
        min_mask_pixels=min_mask_pixels,
        max_masks=max_masks,
        assign_uncovered=False,
        representative_mode=representative_mode,
        mask_pooling_mode=mask_pooling_mode,
    )


def score_clusters(centers, positive_vector, negative_vectors, temperature):
    centers = F.normalize(centers, dim=-1)
    pos = torch.matmul(centers, positive_vector)
    numer = torch.exp(pos * temperature)
    denom = numer.clone()
    neg_scores = []
    for neg in negative_vectors:
        neg_score = torch.matmul(centers, neg)
        neg_scores.append(neg_score)
        denom = denom + torch.exp(neg_score * temperature)
    scores = numer / denom.clamp_min(1e-8)
    return pos, scores


def make_similarity_overlay(image, cluster_id_map, cluster_items):
    image_np = np.asarray(image).astype(np.float32) / 255.0
    score_map = np.zeros_like(cluster_id_map, dtype=np.float32)

    cluster_scores = {}
    for item in cluster_items:
        cluster_id = int(item["cluster_id"])
        cluster_scores[cluster_id] = max(cluster_scores.get(cluster_id, 0.0), float(item["score"]))

    for cluster_id, score in cluster_scores.items():
        score_map[cluster_id_map == cluster_id] = score

    heatmap = cv2.resize(score_map, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
    if np.count_nonzero(heatmap) > 0:
        positive = heatmap[heatmap > 0]
        low = float(np.percentile(positive, 5))
        high = float(np.percentile(positive, 95))
        if high <= low:
            low = float(positive.min())
            high = float(positive.max())
        heatmap = np.clip((heatmap - low) / max(high - low, 1e-8), 0.0, 1.0)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=8, sigmaY=8)

    colored = plt.get_cmap("magma")(heatmap)[..., :3].astype(np.float32)
    alpha = (heatmap ** 0.8) * 0.75
    overlay = image_np * (1.0 - alpha[..., None]) + colored * alpha[..., None]
    return np.clip(overlay, 0.0, 1.0), heatmap


def make_cluster_legend_overlay(image, cluster_id_map, cluster_items, top_clusters):
    image_np = np.asarray(image).astype(np.float32) / 255.0
    top_items = cluster_items[:top_clusters]
    if not top_items:
        return image_np, []

    if cluster_id_map.shape[:2] != image_np.shape[:2]:
        resized_cluster_map = cv2.resize(
            cluster_id_map.astype(np.int32),
            (image.width, image.height),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized_cluster_map = cluster_id_map

    palette = plt.get_cmap("tab10")
    overlay = image_np.copy()
    legend_entries = []
    positive_scores = np.array([max(float(item["score"]), 0.0) for item in top_items], dtype=np.float32)
    max_score = float(positive_scores.max()) if positive_scores.size else 1.0

    for idx, item in enumerate(top_items):
        cluster_id = int(item["cluster_id"])
        score = float(item["score"])
        mask = resized_cluster_map == cluster_id
        if not np.any(mask):
            continue

        color = np.array(palette(idx % 10)[:3], dtype=np.float32)
        alpha = 0.25 + 0.45 * (max(score, 0.0) / max(max_score, 1e-8))
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
        legend_entries.append(
            {
                "cluster_id": cluster_id,
                "score": score,
                "color": color,
            }
        )

    return np.clip(overlay, 0.0, 1.0), legend_entries


def render_demo(
    image_path: Path,
    model_id: str,
    query: str,
    negative_text: str,
    output_path: Path,
    num_clusters: int,
    min_cluster_pixels: int,
    merge_similarity: float,
    segmentation_mode: str,
    subobjects_repo,
    subobject_checkpoint: str,
    subobject_resolution: int,
    subobject_threshold: float,
    subobject_max_tokens: int,
    subobject_crop: int,
    textregion_repo,
    textregion_sam2_checkpoint,
    textregion_model_cfg,
    textregion_points_per_side: int,
    textregion_pred_iou_thresh: float,
    textregion_stability_score_thresh: float,
    textregion_box_nms_thresh: float,
    textregion_max_masks: int,
    region_representative_mode: str,
    mask_pooling_mode: str,
    sam3_repo,
    sam3_checkpoint_path,
    sam3_resolution: int,
    sam3_confidence_threshold: float,
    sam3_max_masks: int,
    temperature: float,
    top_clusters: int,
    anyup_entrypoint: str,
    anyup_use_natten: bool,
    anyup_q_chunk_size,
    anyup_output_size,
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
    hr_features = upsample_anyup_features(
        model,
        upsampler,
        image,
        device,
        q_chunk_size=anyup_q_chunk_size,
        output_size=tuple(anyup_output_size) if anyup_output_size is not None else None,
    )

    if segmentation_mode == "epoc":
        print("Segmenting image with EPOC subobjects...")
        segmenter = load_epoc_segmenter(
            repo_path=subobjects_repo,
            checkpoint=subobject_checkpoint,
            image_resolution=subobject_resolution,
            threshold=subobject_threshold,
            max_tokens=subobject_max_tokens,
            crop=subobject_crop,
            device=device,
        )
        centers, label_map = subobject_feature_map(
            hr_features,
            image,
            segmenter,
            min_mask_pixels=min_cluster_pixels,
        )
    elif segmentation_mode == "textregion":
        print("Segmenting image with TextRegion/SAM2 masks...")
        centers, label_map = textregion_feature_map(
            hr_features,
            image,
            repo_path=textregion_repo,
            model_cfg=textregion_model_cfg,
            checkpoint=textregion_sam2_checkpoint,
            points_per_side=textregion_points_per_side,
            pred_iou_thresh=textregion_pred_iou_thresh,
            stability_score_thresh=textregion_stability_score_thresh,
            box_nms_thresh=textregion_box_nms_thresh,
            max_masks=textregion_max_masks,
            min_mask_pixels=min_cluster_pixels,
            representative_mode=region_representative_mode,
            mask_pooling_mode=mask_pooling_mode,
            device=device,
        )
    elif segmentation_mode == "sam3":
        print("Segmenting image with SAM3 masks...")
        centers, label_map = sam3_feature_map(
            hr_features,
            image,
            query=query,
            repo_path=sam3_repo,
            checkpoint_path=sam3_checkpoint_path,
            resolution=sam3_resolution,
            confidence_threshold=sam3_confidence_threshold,
            max_masks=sam3_max_masks,
            min_mask_pixels=min_cluster_pixels,
            device=device,
            representative_mode=region_representative_mode,
            mask_pooling_mode=mask_pooling_mode,
        )
    else:
        centers, label_map = cluster_feature_map(
            hr_features,
            num_clusters=num_clusters,
            min_cluster_pixels=min_cluster_pixels,
            merge_similarity=merge_similarity,
        )
    print(f"Segmentation produced {centers.shape[0]} regions.")

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

    overlay, heatmap = make_similarity_overlay(image, label_map.detach().cpu().numpy(), cluster_items)
    cluster_overlay, legend_entries = make_cluster_legend_overlay(
        image,
        label_map.detach().cpu().numpy(),
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
    axes[1].set_title(f"Cluster score map\nquery='{query}' vs [{', '.join(negative_prompts)}]")
    axes[1].axis("off")

    axes[2].imshow(cluster_overlay)
    cluster_text = ", ".join(
        f"{item['cluster_id']}:{item['score']:.4f}" for item in cluster_items[:top_clusters]
    )
    axes[2].set_title(
        f"Top cluster overlay + legend\n"
        f"clusters={cluster_text}\n"
        f"{segmentation_mode}, rep={region_representative_mode}, mask={mask_pooling_mode}, regions={centers.shape[0]}"
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
        axes[3].legend(
            handles=handles,
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
        )

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
        output_path = Path("scratch") / f"{image_path.stem}_talk2dino_v2_anyup_cluster_search_{args.query}.png"

    render_demo(
        image_path=image_path,
        model_id=args.model_id,
        query=args.query,
        negative_text=args.negative_text,
        output_path=output_path,
        num_clusters=args.num_clusters,
        min_cluster_pixels=args.min_cluster_pixels,
        merge_similarity=args.merge_similarity,
        segmentation_mode=args.segmentation_mode,
        subobjects_repo=args.subobjects_repo,
        subobject_checkpoint=args.subobject_checkpoint,
        subobject_resolution=args.subobject_resolution,
        subobject_threshold=args.subobject_threshold,
        subobject_max_tokens=args.subobject_max_tokens,
        subobject_crop=args.subobject_crop,
        textregion_repo=args.textregion_repo,
        textregion_sam2_checkpoint=args.textregion_sam2_checkpoint,
        textregion_model_cfg=args.textregion_model_cfg,
        textregion_points_per_side=args.textregion_points_per_side,
        textregion_pred_iou_thresh=args.textregion_pred_iou_thresh,
        textregion_stability_score_thresh=args.textregion_stability_score_thresh,
        textregion_box_nms_thresh=args.textregion_box_nms_thresh,
        textregion_max_masks=args.textregion_max_masks,
        region_representative_mode=args.region_representative_mode,
        mask_pooling_mode=args.mask_pooling_mode,
        sam3_repo=args.sam3_repo,
        sam3_checkpoint_path=args.sam3_checkpoint_path,
        sam3_resolution=args.sam3_resolution,
        sam3_confidence_threshold=args.sam3_confidence_threshold,
        sam3_max_masks=args.sam3_max_masks,
        temperature=args.temperature,
        top_clusters=args.top_clusters,
        anyup_entrypoint=args.anyup_entrypoint,
        anyup_use_natten=args.anyup_use_natten,
        anyup_q_chunk_size=args.anyup_q_chunk_size,
        anyup_output_size=args.anyup_output_size,
    )


if __name__ == "__main__":
    main()
