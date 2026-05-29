import argparse
import csv
import json
import os
import re

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vl_backends import create_backend

ImageFile.LOAD_TRUNCATED_IMAGES = True

METADATA_FIELDS = [
    "final_country",
    "final_city",
    "final_place",
    "date",
    "description",
    "transcription",
    "landmarks_identified",
    "historical_record_uuid",
]


def compact_metadata_text(row):
    def clean(value):
        text = (value or "").strip()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[()]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" ,;")

    priority_values = [
        clean(row.get("final_country")),
        clean(row.get("final_city")),
        clean(row.get("final_place")),
        clean(row.get("landmarks_identified")),
    ]
    description = clean(row.get("description"))[:100]
    transcription = clean(row.get("transcription"))[:60]
    values = [value for value in priority_values + [description, transcription] if value]
    return " ".join(values)[:260]


def load_metadata_texts(metadata_csv):
    if not metadata_csv:
        return {}
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    metadata_by_image = {}
    with open(metadata_csv, "r", encoding="latin1", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = (row.get("image_filename") or "").strip()
            if not image_id:
                continue
            metadata_text = compact_metadata_text(row)
            if metadata_text:
                metadata_by_image[image_id] = metadata_text
    return metadata_by_image


def format_exception(exc):
    text = str(exc)
    return text.encode("ascii", errors="backslashreplace").decode("ascii", errors="replace")


def spherical_kmeans(features, num_clusters=100, num_iters=100, tol=1e-4):
    features = features.float()
    x = F.normalize(features, p=2, dim=-1)
    num_points, feat_dim = x.shape

    if num_points <= num_clusters:
        return x, torch.arange(num_points, device=x.device)

    indices = torch.randperm(num_points, device=x.device)[:num_clusters]
    centers = x[indices]

    for _ in range(num_iters):
        sim = torch.matmul(x, centers.transpose(0, 1))
        labels = torch.argmax(sim, dim=-1)

        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, feat_dim), x)
        new_centers = F.normalize(new_centers, p=2, dim=-1)

        center_shift = (centers * new_centers).sum(dim=-1).mean()
        centers = new_centers
        if 1.0 - center_shift < tol:
            break

    return centers, labels


def _recompute_centers(features, labels, num_clusters):
    feat_dim = features.shape[-1]
    centers = torch.zeros((num_clusters, feat_dim), device=features.device, dtype=features.dtype)
    centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, feat_dim), features)
    counts = torch.bincount(labels, minlength=num_clusters).to(features.device)
    valid = counts > 0
    if valid.any():
        centers[valid] = centers[valid] / counts[valid].unsqueeze(1)
        centers[valid] = F.normalize(centers[valid], p=2, dim=-1)
    return centers, counts


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


def _recompute_geometric_median_centers(features, labels, num_clusters, max_iters=30):
    centers = []
    counts = torch.bincount(labels, minlength=num_clusters).to(features.device)
    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        if mask.any():
            centers.append(geometric_median(features[mask], max_iters=max_iters))
        else:
            centers.append(torch.zeros(features.shape[-1], device=features.device, dtype=features.dtype))
    centers = torch.stack(centers, dim=0) if centers else features[:0]
    return centers, counts


def adaptive_spherical_kmeans(
    features,
    max_clusters=100,
    min_cluster_pixels=8,
    merge_similarity=0.985,
    num_iters=100,
    tol=1e-4,
    representative_mode="mean",
):
    centers, labels = spherical_kmeans(
        features,
        num_clusters=max_clusters,
        num_iters=num_iters,
        tol=tol,
    )

    num_points = features.shape[0]
    if num_points == 0:
        return centers, labels

    centers, counts = _recompute_centers(features, labels, centers.shape[0])
    active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if active_ids.numel() == 0:
        return centers[:0], labels

    if min_cluster_pixels > 1 and active_ids.numel() > 1:
        small_ids = active_ids[counts[active_ids] < min_cluster_pixels]
        large_ids = active_ids[counts[active_ids] >= min_cluster_pixels]
        if small_ids.numel() > 0 and large_ids.numel() > 0:
            for small_id in small_ids.tolist():
                pixel_mask = labels == small_id
                if not pixel_mask.any():
                    continue
                sims = torch.matmul(features[pixel_mask], centers[large_ids].transpose(0, 1))
                best_large = large_ids[torch.argmax(sims, dim=1)]
                labels[pixel_mask] = best_large
            centers, counts = _recompute_centers(features, labels, centers.shape[0])
            active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()

    if merge_similarity is not None and active_ids.numel() > 1:
        while True:
            active_centers = centers[active_ids]
            sim = torch.matmul(active_centers, active_centers.transpose(0, 1))
            sim.fill_diagonal_(-1.0)
            max_sim = torch.max(sim)
            if max_sim < merge_similarity:
                break

            merge_pair = torch.nonzero(sim == max_sim, as_tuple=False)[0]
            left_idx = active_ids[merge_pair[0]].item()
            right_idx = active_ids[merge_pair[1]].item()

            if counts[left_idx] < counts[right_idx]:
                left_idx, right_idx = right_idx, left_idx

            labels[labels == right_idx] = left_idx
            centers, counts = _recompute_centers(features, labels, centers.shape[0])
            active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
            if active_ids.numel() <= 1:
                break

    active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
    remapped_labels = labels.clone()
    final_centers = []
    for new_cluster_id, old_cluster_id in enumerate(active_ids.tolist()):
        remapped_labels[labels == old_cluster_id] = new_cluster_id
        final_centers.append(centers[old_cluster_id])

    if final_centers:
        final_centers = torch.stack(final_centers, dim=0)
    else:
        final_centers = centers[:0]

    if representative_mode == "geometric_median" and final_centers.shape[0] > 0:
        final_centers, _ = _recompute_geometric_median_centers(
            features,
            remapped_labels,
            final_centers.shape[0],
        )
    elif representative_mode != "mean":
        raise ValueError(f"Unsupported representative_mode: {representative_mode}")

    return final_centers, remapped_labels


def downsample_feature_map_for_pooling(feature_map, max_tokens):
    if max_tokens is None or max_tokens <= 0:
        return feature_map

    _, _, height, width = feature_map.shape
    num_tokens = height * width
    if num_tokens <= max_tokens:
        return feature_map

    scale = (max_tokens / float(num_tokens)) ** 0.5
    pooled_height = max(1, int(height * scale))
    pooled_width = max(1, int(width * scale))
    while pooled_height * pooled_width > max_tokens:
        if pooled_height >= pooled_width and pooled_height > 1:
            pooled_height -= 1
        elif pooled_width > 1:
            pooled_width -= 1
        else:
            break
    return F.adaptive_avg_pool2d(feature_map, output_size=(pooled_height, pooled_width))


def assign_to_representatives(features, representatives, chunk_size=8192):
    labels = []
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start : start + chunk_size]
        sims = torch.matmul(chunk, representatives.transpose(0, 1))
        labels.append(torch.argmax(sims, dim=-1))
    return torch.cat(labels, dim=0)


def hierarchical_token_pooling(
    feature_map,
    pool_factor=3,
    max_pool_tokens=4096,
    assignment_chunk_size=8192,
):
    try:
        from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
    except ImportError as exc:
        raise ImportError(
            "Hierarchical pooling requires colpali-engine. Install it with: "
            "pip install colpali-engine"
        ) from exc

    if feature_map.shape[0] != 1:
        raise ValueError("Hierarchical pooling currently expects batch_size=1.")

    pooling_map = downsample_feature_map_for_pooling(feature_map, max_pool_tokens)
    pooling_tokens = pooling_map.permute(0, 2, 3, 1).reshape(-1, pooling_map.shape[1])
    pooling_tokens = F.normalize(pooling_tokens.float(), dim=-1)

    pooler = HierarchicalTokenPooler()
    pooled = pooler.pool_embeddings([pooling_tokens], pool_factor=pool_factor)[0]
    if not isinstance(pooled, torch.Tensor):
        pooled = torch.as_tensor(pooled, device=feature_map.device)
    representatives = F.normalize(pooled.to(feature_map.device).float(), dim=-1)

    dense_flat = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
    dense_flat = F.normalize(dense_flat.float(), dim=-1)
    labels = assign_to_representatives(
        dense_flat,
        representatives,
        chunk_size=assignment_chunk_size,
    )
    return representatives, labels


def slic_feature_clustering(feature_map, num_segments=15, compactness=0.1, representative_mode="mean"):
    try:
        from skimage.segmentation import slic
    except ImportError as exc:
        raise ImportError("SLIC pooling requires scikit-image. Install it with: pip install scikit-image") from exc

    if feature_map.shape[0] != 1:
        raise ValueError("SLIC clustering currently expects batch_size=1.")

    _, channels, height, width = feature_map.shape
    dense_hwc = feature_map[0].permute(1, 2, 0).detach().float().cpu().numpy()
    segments = slic(
        dense_hwc,
        n_segments=int(num_segments),
        compactness=float(compactness),
        channel_axis=-1,
        enforce_connectivity=True,
        start_label=0,
    )
    labels = torch.from_numpy(segments.astype("int64")).to(feature_map.device).reshape(-1)
    num_labels = int(labels.max().item()) + 1 if labels.numel() else 0
    dense_flat = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
    dense_flat = F.normalize(dense_flat.float(), dim=-1)
    if representative_mode == "geometric_median":
        centers, counts = _recompute_geometric_median_centers(dense_flat, labels, num_labels)
    elif representative_mode == "mean":
        centers, counts = _recompute_centers(dense_flat, labels, num_labels)
    else:
        raise ValueError(f"Unsupported representative_mode: {representative_mode}")

    valid_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
    remapped_labels = labels.clone()
    final_centers = []
    for new_id, old_id in enumerate(valid_ids.tolist()):
        remapped_labels[labels == old_id] = new_id
        final_centers.append(centers[old_id])
    if final_centers:
        final_centers = torch.stack(final_centers, dim=0)
    else:
        final_centers = centers[:0]
    return final_centers, remapped_labels


class FeatureBatchExtractor:
    def __init__(
        self,
        backend_name="tips",
        device="cuda",
        num_clusters=100,
        min_cluster_pixels=8,
        merge_similarity=0.985,
        model_version="c-radio_v4-h",
        lang_model="siglip2-g",
        model_id=None,
        anyup_entrypoint="anyup_multi_backbone",
        anyup_use_natten=False,
        anyup_q_chunk_size=None,
        anyup_output_size=(384, 384),
        metadata_by_image=None,
        pooling_mode="kmeans",
        pool_factor=3,
        hierarchical_max_tokens=4096,
        assignment_chunk_size=8192,
        slic_compactness=0.1,
        representative_mode="mean",
    ):
        self.device = device
        self.num_clusters = num_clusters
        self.min_cluster_pixels = min_cluster_pixels
        self.merge_similarity = merge_similarity
        self.backend_name = backend_name
        self.metadata_by_image = metadata_by_image or {}
        self.pooling_mode = pooling_mode
        self.pool_factor = pool_factor
        self.hierarchical_max_tokens = hierarchical_max_tokens
        self.assignment_chunk_size = assignment_chunk_size
        self.slic_compactness = slic_compactness
        self.representative_mode = representative_mode

        self.backend = create_backend(
            backend_name=backend_name,
            device=self.device,
            model_version=model_version,
            lang_model=lang_model,
            model_id=model_id,
            anyup_entrypoint=anyup_entrypoint,
            anyup_use_natten=anyup_use_natten,
            anyup_q_chunk_size=anyup_q_chunk_size,
            anyup_output_size=anyup_output_size,
        )
        self.transform = self.backend.transform

    @torch.no_grad()
    def process_tensor(self, img_tensor, image_id=None):
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.to(self.device)
        visual_aligned = self.backend.encode_image_to_feature_map(img_tensor)
        image_embedding = self.backend.encode_image_embedding(img_tensor, feature_map=visual_aligned)

        _, channels, height_fm, width_fm = visual_aligned.shape
        dense_flat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, channels)
        dense_flat = F.normalize(dense_flat, dim=-1)

        if self.pooling_mode == "kmeans":
            centers, labels = adaptive_spherical_kmeans(
                dense_flat,
                max_clusters=self.num_clusters,
                min_cluster_pixels=self.min_cluster_pixels,
                merge_similarity=self.merge_similarity,
                representative_mode=self.representative_mode,
            )
        elif self.pooling_mode == "hierarchical":
            centers, labels = hierarchical_token_pooling(
                visual_aligned,
                pool_factor=self.pool_factor,
                max_pool_tokens=self.hierarchical_max_tokens,
                assignment_chunk_size=self.assignment_chunk_size,
            )
        elif self.pooling_mode == "slic":
            centers, labels = slic_feature_clustering(
                visual_aligned,
                num_segments=self.num_clusters,
                compactness=self.slic_compactness,
                representative_mode=self.representative_mode,
            )
        else:
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")
        label_map = labels.reshape(height_fm, width_fm)

        clusters = [
            {
                "cluster_id": int(cluster_idx),
                "v": centers[cluster_idx].detach().cpu().tolist(),
            }
            for cluster_idx in range(centers.shape[0])
        ]

        result = {
            "clusters": clusters,
            "feature_map_size": [int(height_fm), int(width_fm)],
            "cluster_id_map": label_map.detach().cpu().tolist(),
            "pooling_mode": self.pooling_mode,
            "representative_mode": self.representative_mode,
        }
        if self.pooling_mode == "hierarchical":
            result["pool_factor"] = int(self.pool_factor)
            result["hierarchical_max_tokens"] = int(self.hierarchical_max_tokens)
        if self.pooling_mode == "slic":
            result["slic_compactness"] = float(self.slic_compactness)
        if image_embedding is not None:
            result["image_embedding"] = image_embedding.squeeze(0).detach().cpu().tolist()
            result["image_embedding_type"] = "cls"
        if image_id and image_id in self.metadata_by_image:
            metadata_embedding = self.backend.encode_text([self.metadata_by_image[image_id]])
            result["metadata_embedding"] = metadata_embedding.squeeze(0).detach().cpu().tolist()
            result["metadata_text"] = self.metadata_by_image[image_id]

        return result


class FastImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_name = os.path.basename(path)
        try:
            img = Image.open(path).convert("RGB")
            sample = self.transform(img) if self.transform is not None else img
            return sample, img_name, True
        except Exception:
            fallback = torch.zeros((3, 448, 448)) if self.transform is not None else None
            return fallback, img_name, False


def passthrough_collate(batch):
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch extract clustered dense vision-language features from images.")
    parser.add_argument("--input_dir", type=str, default="images/", help="Directory with images")
    parser.add_argument("--output_file", type=str, default="tmp/features.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--backend",
        type=str,
        default="tips",
        choices=["tips", "talk2dino", "talk2dino_anyup", "radseg"],
        help="Vision-language backend used to produce dense features",
    )
    parser.add_argument(
        "--start_image",
        type=int,
        default=1,
        help="1-based inclusive start index of images to process after sorting",
    )
    parser.add_argument(
        "--end_image",
        type=int,
        default=None,
        help="1-based inclusive end index of images to process after sorting",
    )
    parser.add_argument("--num_clusters", type=int, default=100, help="Maximum number of clusters per image")
    parser.add_argument(
        "--pooling_mode",
        choices=["kmeans", "hierarchical", "slic"],
        default="kmeans",
        help="How to create region representatives from dense patch embeddings",
    )
    parser.add_argument(
        "--pool_factor",
        type=int,
        default=3,
        help="HierarchicalTokenPooler compression factor when --pooling_mode=hierarchical",
    )
    parser.add_argument(
        "--hierarchical_max_tokens",
        type=int,
        default=4096,
        help="Downsample dense maps to at most this many tokens before hierarchical pooling; set <=0 to disable",
    )
    parser.add_argument(
        "--assignment_chunk_size",
        type=int,
        default=8192,
        help="Chunk size for assigning full-resolution dense tokens back to representatives",
    )
    parser.add_argument("--slic_compactness", type=float, default=0.1, help="SLIC compactness when --pooling_mode=slic")
    parser.add_argument(
        "--representative_mode",
        choices=["mean", "geometric_median"],
        default="mean",
        help="How to compute each cluster representative after clustering",
    )
    parser.add_argument(
        "--min_cluster_pixels",
        type=int,
        default=8,
        help="Clusters smaller than this are reassigned to larger clusters",
    )
    parser.add_argument(
        "--merge_similarity",
        type=float,
        default=0.985,
        help="Merge clusters whose cosine similarity exceeds this threshold",
    )
    parser.add_argument("--model_version", type=str, default="c-radio_v4-h")
    parser.add_argument("--lang_model", type=str, default="siglip2-g")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--metadata_csv", type=str, default=None, help="Optional metadata CSV used to add metadata embeddings")
    parser.add_argument("--anyup_entrypoint", type=str, default="anyup_multi_backbone")
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=None)
    parser.add_argument(
        "--anyup_output_size",
        nargs=2,
        type=int,
        default=[384, 384],
        metavar=("WIDTH", "HEIGHT"),
        help="AnyUp output size when backend=talk2dino_anyup",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    metadata_by_image = load_metadata_texts(args.metadata_csv)
    if metadata_by_image:
        print(f"Loaded metadata text for {len(metadata_by_image)} images.")

    extractor = FeatureBatchExtractor(
        backend_name=args.backend,
        num_clusters=args.num_clusters,
        model_version=args.model_version,
        lang_model=args.lang_model,
        min_cluster_pixels=args.min_cluster_pixels,
        merge_similarity=args.merge_similarity,
        model_id=args.model_id,
        device=args.device,
        anyup_entrypoint=args.anyup_entrypoint,
        anyup_use_natten=args.anyup_use_natten,
        anyup_q_chunk_size=args.anyup_q_chunk_size,
        anyup_output_size=tuple(args.anyup_output_size) if args.anyup_output_size is not None else None,
        metadata_by_image=metadata_by_image,
        pooling_mode=args.pooling_mode,
        pool_factor=args.pool_factor,
        hierarchical_max_tokens=args.hierarchical_max_tokens,
        assignment_chunk_size=args.assignment_chunk_size,
        slic_compactness=args.slic_compactness,
        representative_mode=args.representative_mode,
    )

    if not os.path.exists(args.input_dir):
        print(f"Error: {args.input_dir} not found.")
        raise SystemExit(1)

    image_paths = sorted(
        [
            os.path.join(args.input_dir, name)
            for name in os.listdir(args.input_dir)
            if name.casefold().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    print(f"Found {len(image_paths)} images in {args.input_dir}")

    total_images = len(image_paths)
    if args.start_image < 1:
        print("Error: --start_image must be >= 1.")
        raise SystemExit(1)
    if args.end_image is not None and args.end_image < args.start_image:
        print("Error: --end_image must be >= --start_image.")
        raise SystemExit(1)

    start_idx = args.start_image - 1
    end_idx = total_images if args.end_image is None else min(args.end_image, total_images)
    if start_idx >= total_images:
        print(f"Error: --start_image={args.start_image} exceeds total images {total_images}.")
        raise SystemExit(1)

    image_paths = image_paths[start_idx:end_idx]
    print(
        f"Selected images {args.start_image} to {start_idx + len(image_paths)} "
        f"(count={len(image_paths)}) for this run."
    )

    dataset = FastImageDataset(image_paths, extractor.transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        collate_fn=passthrough_collate,
    )

    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as read_file:
            for line in read_file:
                try:
                    processed_ids.add(json.loads(line)["image_id"])
                except Exception:
                    continue
        print(f"Resuming: {len(processed_ids)} images already processed. Skipping them.")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "a", encoding="utf-8") as write_file:
        for batch in tqdm(dataloader):
            sample, img_name, is_valid = batch[0]

            if not is_valid:
                print(f"Skipping unreadable image: {img_name}")
                continue
            if img_name in processed_ids:
                continue

            try:
                result = extractor.process_tensor(
                    sample.unsqueeze(0) if isinstance(sample, torch.Tensor) else sample,
                    image_id=img_name,
                )
                result["image_id"] = img_name
                write_file.write(json.dumps(result) + "\n")
                write_file.flush()
            except Exception as exc:
                print(f"Error processing {img_name}: {format_exception(exc)}")
