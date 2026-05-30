import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from batch_extract_features import (
    FastImageDataset,
    format_exception,
    load_metadata_texts,
    passthrough_collate,
)
from demo_talk2dino_v2_anyup_cluster_search import (
    feature_regions_from_masks,
    load_epoc_segmenter,
    load_textregion_mask_generator,
    subobject_feature_map,
)
from vl_backends import create_backend

ImageFile.LOAD_TRUNCATED_IMAGES = True


def safe_encode_metadata(backend, metadata_text):
    text = (metadata_text or "").strip()
    if not text:
        return None, None

    words = text.split()
    candidates = [
        text,
        " ".join(words[:32]),
        " ".join(words[:24]),
        " ".join(words[:16]),
        text[:180],
        text[:140],
        text[:100],
    ]
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            embedding = backend.encode_text([candidate])
            return embedding.squeeze(0).detach().cpu().tolist(), candidate
        except Exception:
            continue
    return None, None


@torch.no_grad()
def extract_region_record(
    backend,
    segmentation_mode,
    textregion_generator,
    subobject_segmenter,
    image,
    image_id,
    metadata_by_image,
    textregion_max_masks,
    min_cluster_pixels,
    representative_mode,
    mask_pooling_mode,
    device,
):
    feature_map = backend.encode_image_to_feature_map(image)
    image_embedding = backend.encode_image_embedding(image, feature_map=feature_map)

    if segmentation_mode == "subobject":
        centers, label_map = subobject_feature_map(
            feature_map,
            image,
            subobject_segmenter,
            min_mask_pixels=min_cluster_pixels,
        )
    elif segmentation_mode == "textregion":
        image_np = torch.as_tensor(
            __import__("numpy").asarray(image.convert("RGB")).copy(),
            device=device,
            dtype=torch.float32,
        )
        image_tensor_for_sam2 = torch.stack([image_np])
        image_tensor_for_sam2 = textregion_generator.predictor._transforms(image_tensor_for_sam2)
        ori_shape = image.size[1], image.size[0]
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device == "cuda"):
            sam2_masks = textregion_generator.generate_for_batch(image_tensor_for_sam2, [ori_shape], None)

        if sam2_masks and sam2_masks[0]:
            masks = torch.stack([mask["segmentations"] for mask in sam2_masks[0]])
            centers, label_map = feature_regions_from_masks(
                feature_map,
                masks,
                min_mask_pixels=min_cluster_pixels,
                max_masks=textregion_max_masks,
                representative_mode=representative_mode,
                mask_pooling_mode=mask_pooling_mode,
            )
        else:
            _, channels, height_fm, width_fm = feature_map.shape
            flat = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
            center = torch.nn.functional.normalize(flat.mean(dim=0, keepdim=True), dim=-1)
            centers = center
            label_map = torch.zeros((height_fm, width_fm), device=feature_map.device, dtype=torch.long)
    else:
        raise ValueError(f"Unsupported segmentation_mode: {segmentation_mode}")

    clusters = [
        {
            "cluster_id": int(cluster_idx),
            "v": centers[cluster_idx].detach().cpu().tolist(),
        }
        for cluster_idx in range(centers.shape[0])
    ]

    result = {
        "image_id": image_id,
        "clusters": clusters,
        "feature_map_size": [int(label_map.shape[0]), int(label_map.shape[1])],
        "cluster_id_map": label_map.detach().cpu().tolist(),
        "pooling_mode": "subobject_epoc" if segmentation_mode == "subobject" else "textregion_sam2",
        "representative_mode": representative_mode,
        "mask_pooling_mode": mask_pooling_mode,
        "segmentation_mode": segmentation_mode,
    }
    if segmentation_mode == "textregion":
        result["textregion_max_masks"] = int(textregion_max_masks)

    if image_embedding is not None:
        result["image_embedding"] = image_embedding.squeeze(0).detach().cpu().tolist()
        result["image_embedding_type"] = getattr(
            backend,
            "image_embedding_type",
            "mean_pooled_dense",
        )

    metadata_text = metadata_by_image.get(image_id)
    if metadata_text:
        metadata_embedding, encoded_text = safe_encode_metadata(backend, metadata_text)
        if metadata_embedding is not None:
            result["metadata_embedding"] = metadata_embedding
            result["metadata_text"] = encoded_text

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract AnyUp TextRegion/SAM2 region vectors with global and metadata embeddings."
    )
    parser.add_argument("--input_dir", type=str, default="images")
    parser.add_argument("--output_file", type=str, default="features_talk2dinov3_anyup_textregion_geom_1_1000.jsonl")
    parser.add_argument("--start_image", type=int, default=1)
    parser.add_argument("--end_image", type=int, default=1000)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["tips_anyup", "talk2dino_anyup"],
        default="talk2dino_anyup",
        help="AnyUp backend used before TextRegion/SAM2 region pooling.",
    )
    parser.add_argument("--model_id", type=str, default="lorebianchi98/Talk2DINOv3-ViTL")
    parser.add_argument("--metadata_csv", type=str, default="images_metadata.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--anyup_entrypoint", type=str, default="anyup_multi_backbone")
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=10)
    parser.add_argument("--anyup_output_size", nargs=2, type=int, default=[384, 384])
    parser.add_argument(
        "--segmentation_mode",
        choices=["textregion", "subobject"],
        default="textregion",
        help="Region source after AnyUp: TextRegion/SAM2 masks or EPOC subobjects.",
    )
    parser.add_argument("--textregion_repo", type=str, default="scratch/TextRegion")
    parser.add_argument("--textregion_sam2_checkpoint", type=str, default="scratch/TextRegion/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--textregion_model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--textregion_points_per_side", type=int, default=16)
    parser.add_argument("--textregion_max_masks", type=int, default=32)
    parser.add_argument("--textregion_pred_iou_thresh", type=float, default=0.6)
    parser.add_argument("--textregion_stability_score_thresh", type=float, default=0.6)
    parser.add_argument("--textregion_box_nms_thresh", type=float, default=0.9)
    parser.add_argument(
        "--subobjects_repo",
        type=str,
        default=None,
        help="Path to a local clone of https://github.com/ChenDelong1999/subobjects.",
    )
    parser.add_argument(
        "--subobject_checkpoint",
        type=str,
        default="chendelong/DirectSAM-b0-1024px-sa1b-2ep-1017",
        help="DirectSAM checkpoint for EPOC subobject segmentation.",
    )
    parser.add_argument("--subobject_resolution", type=int, default=1024)
    parser.add_argument("--subobject_threshold", type=float, default=0.1)
    parser.add_argument("--subobject_max_tokens", type=int, default=64)
    parser.add_argument("--subobject_crop", type=int, default=1)
    parser.add_argument("--min_cluster_pixels", type=int, default=8)
    parser.add_argument("--representative_mode", choices=["mean", "geometric_median"], default="geometric_median")
    parser.add_argument("--mask_pooling_mode", choices=["hard", "soft"], default="hard")
    args = parser.parse_args()

    metadata_by_image = load_metadata_texts(args.metadata_csv)
    if metadata_by_image:
        print(f"Loaded metadata text for {len(metadata_by_image)} images.")

    backend = create_backend(
        backend_name=args.backend,
        device=args.device,
        model_id=args.model_id,
        anyup_entrypoint=args.anyup_entrypoint,
        anyup_use_natten=args.anyup_use_natten,
        anyup_q_chunk_size=args.anyup_q_chunk_size,
        anyup_output_size=tuple(args.anyup_output_size) if args.anyup_output_size is not None else None,
    )
    textregion_generator = None
    subobject_segmenter = None
    if args.segmentation_mode == "textregion":
        textregion_generator = load_textregion_mask_generator(
            repo_path=args.textregion_repo,
            model_cfg=args.textregion_model_cfg,
            checkpoint=args.textregion_sam2_checkpoint,
            points_per_side=args.textregion_points_per_side,
            pred_iou_thresh=args.textregion_pred_iou_thresh,
            stability_score_thresh=args.textregion_stability_score_thresh,
            box_nms_thresh=args.textregion_box_nms_thresh,
            device=args.device,
        )
    elif args.segmentation_mode == "subobject":
        subobject_segmenter = load_epoc_segmenter(
            repo_path=args.subobjects_repo,
            checkpoint=args.subobject_checkpoint,
            image_resolution=args.subobject_resolution,
            threshold=args.subobject_threshold,
            max_tokens=args.subobject_max_tokens,
            crop=args.subobject_crop,
            device=args.device,
        )

    input_dir = Path(args.input_dir)
    image_paths = sorted(
        [
            str(input_dir / name)
            for name in os.listdir(input_dir)
            if name.casefold().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    total_images = len(image_paths)
    start_idx = max(args.start_image - 1, 0)
    end_idx = total_images if args.end_image is None else min(args.end_image, total_images)
    image_paths = image_paths[start_idx:end_idx]
    print(f"Selected images {args.start_image} to {start_idx + len(image_paths)} (count={len(image_paths)}).")

    dataset = FastImageDataset(image_paths, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False, shuffle=False, collate_fn=passthrough_collate)

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

    progress_label = f"Extracting {args.segmentation_mode} vectors"
    with open(args.output_file, "a", encoding="utf-8") as write_file:
        for batch in tqdm(dataloader, desc=progress_label):
            image, image_id, is_valid = batch[0]
            if not is_valid:
                print(f"Skipping unreadable image: {image_id}")
                continue
            if image_id in processed_ids:
                continue
            try:
                result = extract_region_record(
                    backend=backend,
                    segmentation_mode=args.segmentation_mode,
                    textregion_generator=textregion_generator,
                    subobject_segmenter=subobject_segmenter,
                    image=image,
                    image_id=image_id,
                    metadata_by_image=metadata_by_image,
                    textregion_max_masks=args.textregion_max_masks,
                    min_cluster_pixels=args.min_cluster_pixels,
                    representative_mode=args.representative_mode,
                    mask_pooling_mode=args.mask_pooling_mode,
                    device=args.device,
                )
                write_file.write(json.dumps(result) + "\n")
                write_file.flush()
            except Exception as exc:
                print(f"Error processing {image_id}: {format_exception(exc)}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
