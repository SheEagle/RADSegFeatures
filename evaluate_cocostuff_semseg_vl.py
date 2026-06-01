import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm

from vl_backends import create_backend


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate open-vocabulary dense VL backends on COCO-Stuff semantic masks."
    )
    parser.add_argument("--data_root", default="datasets/coco_stuff164k")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--class_file", default="evaluation/2d/configs/cls_coco_stuff.txt")
    parser.add_argument(
        "--raw_label_file",
        default=None,
        help=(
            "Optional COCO-Stuff labels.txt mapping raw PNG ids to names. "
            "Defaults to DATA_ROOT/labels.txt when present."
        ),
    )
    parser.add_argument("--output_dir", default="scratch/semseg_cocostuff")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--start_image", type=int, default=0)
    parser.add_argument("--mask_suffix", default=".png")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument(
        "--label_offset",
        choices=("auto", "zero_based", "one_based"),
        default="auto",
        help="Use one_based for masks whose class ids are 1..N and 0/255 are ignored.",
    )
    parser.add_argument(
        "--prompt_template",
        default="{}",
        help="Template applied to each class prompt, e.g. 'a photo of {}'.",
    )
    parser.add_argument(
        "--prompt_space_names",
        action="store_true",
        help="Replace '-' and '_' with spaces before text encoding.",
    )
    parser.add_argument("--save_visualizations", type=int, default=0)

    parser.add_argument(
        "--backend",
        choices=("radseg", "tips", "tips_anyup", "talk2dino", "talk2dino_anyup"),
        required=True,
    )
    parser.add_argument("--model_version", default="c-radio_v4-h")
    parser.add_argument("--lang_model", default="siglip2-g")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--anyup_entrypoint", default="anyup_multi_backbone")
    parser.add_argument("--anyup_use_natten", action="store_true")
    parser.add_argument("--anyup_q_chunk_size", type=int, default=None)
    parser.add_argument("--anyup_output_size", type=int, nargs=2, default=(384, 384))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_classes(class_file, prompt_space_names=False, prompt_template="{}"):
    labels = []
    prompts = []
    with open(class_file, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            # Some RADSeg class files use comma-separated aliases. Use the first
            # name as the metric label and encode all aliases only if requested
            # in the file through separate lines.
            label = raw.split(", ")[0]
            prompt_name = label
            if prompt_space_names:
                prompt_name = prompt_name.replace("-", " ").replace("_", " ")
            labels.append(label)
            prompts.append(prompt_template.format(prompt_name))
    if not labels:
        raise ValueError(f"No class names found in {class_file}")
    return labels, prompts


def normalize_label_name(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def read_raw_label_map(raw_label_file, class_labels):
    if raw_label_file is None:
        return None
    raw_label_path = Path(raw_label_file)
    if not raw_label_path.exists():
        return None

    text = raw_label_path.read_text(encoding="utf-8").strip()
    raw_pairs = re.findall(r"(\d+)\s+([^\d]+?)(?=\s+\d+\s+|$)", text)
    if not raw_pairs:
        raise ValueError(f"Could not parse raw COCO-Stuff label file: {raw_label_file}")

    compact_by_name = {
        normalize_label_name(label): idx for idx, label in enumerate(class_labels)
    }
    raw_to_compact = {}
    missing_names = []
    for raw_id_text, raw_name in raw_pairs:
        raw_id = int(raw_id_text)
        raw_name = raw_name.strip()
        compact_id = compact_by_name.get(normalize_label_name(raw_name))
        if compact_id is None:
            missing_names.append(raw_name)
            continue
        raw_to_compact[raw_id] = compact_id

    if not raw_to_compact:
        raise ValueError(f"No raw COCO-Stuff labels matched class file labels: {raw_label_file}")
    return {
        "path": str(raw_label_path),
        "raw_to_compact": raw_to_compact,
        "missing_names": missing_names,
    }


def resolve_dirs(args):
    if args.image_dir is not None:
        image_dir = Path(args.image_dir)
    else:
        image_dir = Path(args.data_root) / "images" / args.split
    if args.mask_dir is not None:
        mask_dir = Path(args.mask_dir)
    else:
        mask_dir = Path(args.data_root) / "annotations" / args.split
    return image_dir, mask_dir


def discover_pairs(image_dir, mask_dir, mask_suffix):
    image_paths_by_name = {}
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for image_path in image_dir.glob(suffix):
            image_paths_by_name[image_path.name.lower()] = image_path
    pairs = []
    for image_path in sorted(image_paths_by_name.values()):
        mask_path = mask_dir / f"{image_path.stem}{mask_suffix}"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def infer_offset(mask_np, num_classes, ignore_index):
    valid = mask_np != ignore_index
    values = mask_np[valid]
    if values.size == 0:
        return "zero_based"
    min_value = int(values.min())
    max_value = int(values.max())
    if max_value == num_classes or (min_value >= 1 and max_value <= num_classes):
        return "one_based"
    return "zero_based"


def remap_mask(mask_np, num_classes, ignore_index, label_offset, raw_label_map=None):
    if raw_label_map is not None:
        raw_to_compact = raw_label_map["raw_to_compact"]
        remapped = np.full(mask_np.shape, ignore_index, dtype=np.int64)
        for raw_id, compact_id in raw_to_compact.items():
            remapped[mask_np == raw_id] = compact_id
        return remapped, "raw_id_map"

    mode = infer_offset(mask_np, num_classes, ignore_index) if label_offset == "auto" else label_offset
    remapped = np.full(mask_np.shape, ignore_index, dtype=np.int64)
    base_valid = mask_np != ignore_index
    if mode == "one_based":
        valid = base_valid & (mask_np >= 1) & (mask_np <= num_classes)
        remapped[valid] = mask_np[valid].astype(np.int64) - 1
    else:
        valid = base_valid & (mask_np >= 0) & (mask_np < num_classes)
        remapped[valid] = mask_np[valid].astype(np.int64)
    return remapped, mode


def prepare_image_for_backend(backend, pil_image, device):
    if backend.transform is None:
        return pil_image
    return backend.transform(pil_image).unsqueeze(0).to(device)


def update_confusion(confusion, pred, target, num_classes, ignore_index):
    pred = pred.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    if valid.sum() == 0:
        return confusion
    indices = target[valid] * num_classes + pred[valid].clamp(0, num_classes - 1)
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes).cpu()
    return confusion


def metrics_from_confusion(confusion):
    confusion = confusion.to(torch.float64)
    true_positive = torch.diag(confusion)
    gt_count = confusion.sum(dim=1)
    pred_count = confusion.sum(dim=0)
    union = gt_count + pred_count - true_positive

    valid_iou = union > 0
    iou = torch.full_like(union, float("nan"))
    iou[valid_iou] = true_positive[valid_iou] / union[valid_iou]

    valid_acc = gt_count > 0
    class_acc = torch.full_like(gt_count, float("nan"))
    class_acc[valid_acc] = true_positive[valid_acc] / gt_count[valid_acc]

    total = confusion.sum()
    pixel_acc = true_positive.sum() / total if total > 0 else torch.tensor(float("nan"))
    miou = torch.nanmean(iou) if valid_iou.any() else torch.tensor(float("nan"))
    mean_acc = torch.nanmean(class_acc) if valid_acc.any() else torch.tensor(float("nan"))
    return {
        "iou": iou,
        "class_acc": class_acc,
        "pixel_acc": float(pixel_acc),
        "miou": float(miou),
        "mean_acc": float(mean_acc),
    }


def colorize_mask(mask, palette, ignore_index):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != ignore_index
    if valid.any():
        rgb[valid] = palette[mask[valid] % len(palette)]
    return Image.fromarray(rgb)


def build_palette(num_classes):
    rng = np.random.default_rng(12345)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[:8] = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
        ],
        dtype=np.uint8,
    )
    return palette


def save_visualization(image, pred, target, palette, ignore_index, out_path):
    image = image.convert("RGB")
    gt_img = colorize_mask(target, palette, ignore_index).resize(image.size, Image.Resampling.NEAREST)
    pred_img = colorize_mask(pred, palette, ignore_index).resize(image.size, Image.Resampling.NEAREST)

    width, height = image.size
    canvas = Image.new("RGB", (width * 3, height), "white")
    canvas.paste(image, (0, 0))
    canvas.paste(gt_img, (width, 0))
    canvas.paste(pred_img, (width * 2, 0))
    canvas.save(out_path)


@torch.no_grad()
def predict_mask(backend, image, text_features, target_size, device):
    model_input = prepare_image_for_backend(backend, image, device)
    feature_map = backend.encode_image_to_feature_map(model_input)
    feature_map = F.normalize(feature_map, dim=1)
    scores = torch.einsum("bchw,nc->bnhw", feature_map, text_features)
    scores = F.interpolate(scores, size=target_size, mode="bilinear", align_corners=False)
    return scores.argmax(dim=1).squeeze(0).cpu()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    if args.save_visualizations > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_dir, mask_dir = resolve_dirs(args)
    if args.raw_label_file is None:
        candidate_raw_label_file = Path(args.data_root) / "labels.txt"
        args.raw_label_file = str(candidate_raw_label_file) if candidate_raw_label_file.exists() else None
    labels, prompts = read_classes(
        args.class_file,
        prompt_space_names=args.prompt_space_names,
        prompt_template=args.prompt_template,
    )
    num_classes = len(labels)
    raw_label_map = read_raw_label_map(args.raw_label_file, labels)

    pairs = discover_pairs(image_dir, mask_dir, args.mask_suffix)
    if args.start_image:
        pairs = pairs[args.start_image :]
    if args.max_images is not None:
        pairs = pairs[: args.max_images]
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found. image_dir={image_dir}, mask_dir={mask_dir}, mask_suffix={args.mask_suffix}"
        )

    device = args.device
    backend = create_backend(
        backend_name=args.backend,
        device=device,
        model_version=args.model_version,
        lang_model=args.lang_model,
        model_id=args.model_id,
        anyup_entrypoint=args.anyup_entrypoint,
        anyup_use_natten=args.anyup_use_natten,
        anyup_q_chunk_size=args.anyup_q_chunk_size,
        anyup_output_size=tuple(args.anyup_output_size),
    )
    text_features = backend.encode_text(prompts).to(device)
    text_features = F.normalize(text_features, dim=-1)

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    image_rows = []
    offset_counts = {}
    palette = build_palette(num_classes)

    for idx, (image_path, mask_path) in enumerate(tqdm(pairs, desc="Evaluating COCO-Stuff")):
        image = Image.open(image_path).convert("RGB")
        mask_np = np.array(Image.open(mask_path))
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        target_np, offset_mode = remap_mask(
            mask_np,
            num_classes,
            args.ignore_index,
            args.label_offset,
            raw_label_map=raw_label_map,
        )
        offset_counts[offset_mode] = offset_counts.get(offset_mode, 0) + 1

        target = torch.from_numpy(target_np)
        pred = predict_mask(backend, image, text_features, target.shape, device)

        image_confusion = torch.zeros_like(confusion)
        image_confusion = update_confusion(
            image_confusion,
            pred,
            target,
            num_classes,
            args.ignore_index,
        )
        confusion = update_confusion(confusion, pred, target, num_classes, args.ignore_index)
        image_metrics = metrics_from_confusion(image_confusion)
        image_rows.append(
            {
                "image": image_path.name,
                "mask": mask_path.name,
                "miou": image_metrics["miou"],
                "pixel_acc": image_metrics["pixel_acc"],
                "valid_pixels": int((target != args.ignore_index).sum().item()),
                "label_offset": offset_mode,
            }
        )

        if idx < args.save_visualizations:
            save_visualization(
                image,
                pred.numpy().astype(np.int64),
                target_np,
                palette,
                args.ignore_index,
                vis_dir / f"{idx:04d}_{image_path.stem}.png",
            )

    metrics = metrics_from_confusion(confusion)
    summary = {
        "backend": args.backend,
        "model_version": args.model_version,
        "lang_model": args.lang_model,
        "model_id": args.model_id,
        "num_images": len(pairs),
        "num_classes": num_classes,
        "mIoU": metrics["miou"],
        "mean_accuracy": metrics["mean_acc"],
        "pixel_accuracy": metrics["pixel_acc"],
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "class_file": args.class_file,
        "label_offset_counts": offset_counts,
        "prompt_template": args.prompt_template,
        "prompt_space_names": args.prompt_space_names,
        "raw_label_file": args.raw_label_file,
        "raw_label_mapping": {
            "enabled": raw_label_map is not None,
            "mapped_raw_labels": len(raw_label_map["raw_to_compact"]) if raw_label_map else 0,
            "ignored_raw_labels": raw_label_map["missing_names"] if raw_label_map else [],
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(output_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "mask", "miou", "pixel_acc", "valid_pixels", "label_offset"],
        )
        writer.writeheader()
        writer.writerows(image_rows)

    with open(output_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "class_name", "iou", "class_acc", "gt_pixels", "pred_pixels"],
        )
        writer.writeheader()
        ious = metrics["iou"].tolist()
        class_accs = metrics["class_acc"].tolist()
        for class_id, class_name in enumerate(labels):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": ious[class_id],
                    "class_acc": class_accs[class_id],
                    "gt_pixels": int(confusion[class_id].sum().item()),
                    "pred_pixels": int(confusion[:, class_id].sum().item()),
                }
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
