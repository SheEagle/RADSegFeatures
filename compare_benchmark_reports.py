import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd


def load_summary(report_dir):
    report_dir = Path(report_dir)
    with (report_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    image_df = pd.read_csv(report_dir / "image_level_metrics.csv")
    cluster_df = pd.read_csv(report_dir / "cluster_level_metrics.csv")
    gt_path = report_dir / "query_ground_truth.json"
    if gt_path.exists():
        with gt_path.open("r", encoding="utf-8") as handle:
            ground_truth = json.load(handle)
    else:
        ground_truth = []
    return report_dir, summary, image_df, cluster_df, ground_truth


def fmt(value):
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def slugify_for_filename(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "benchmark"


def family_table(summary_a, summary_b, family_key, title):
    rows = []
    families = sorted(
        set(summary_a.get(family_key, {}).keys()) | set(summary_b.get(family_key, {}).keys())
    )
    if not families:
        return ""

    metric_order = {
        "image_level_family_macro": ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"],
        "cluster_level_family_macro": [
            "cluster_precision@10",
            "cluster_precision@20",
            "image_recall_from_clusters@10",
            "image_recall_from_clusters@20",
            "cluster_mrr",
        ],
    }[family_key]

    rows.append(f"<h2>{html.escape(title)}</h2>")
    rows.append("<table>")
    rows.append("<tr><th>Family</th><th>Metric</th><th>RADSeg</th><th>TIPS</th></tr>")
    for family in families:
        for idx, metric in enumerate(metric_order):
            a_val = summary_a.get(family_key, {}).get(family, {}).get(metric)
            b_val = summary_b.get(family_key, {}).get(family, {}).get(metric)
            family_cell = html.escape(family) if idx == 0 else ""
            rows.append(
                f"<tr><td>{family_cell}</td><td>{html.escape(metric)}</td><td>{fmt(a_val)}</td><td>{fmt(b_val)}</td></tr>"
            )
    rows.append("</table>")
    return "".join(rows)


def per_query_table(image_a, image_b, cluster_a, cluster_b):
    a = image_a.merge(
        cluster_a[
            [
                "benchmark_type",
                "query",
                "cluster_precision@10",
                "cluster_precision@20",
                "cluster_mrr",
            ]
        ],
        on=["benchmark_type", "query"],
        how="left",
    )
    b = image_b.merge(
        cluster_b[
            [
                "benchmark_type",
                "query",
                "cluster_precision@10",
                "cluster_precision@20",
                "cluster_mrr",
            ]
        ],
        on=["benchmark_type", "query"],
        how="left",
    )
    merged = a.merge(
        b,
        on=["benchmark_type", "query"],
        how="outer",
        suffixes=("_radseg", "_tips"),
    ).sort_values(["benchmark_type", "query"])

    rows = [
        "<h2>Per-query Side-by-side</h2>",
        "<table>",
        "<tr><th>Type</th><th>Query</th><th>RADSeg image MRR</th><th>TIPS image MRR</th><th>RADSeg Recall@10</th><th>TIPS Recall@10</th><th>RADSeg cluster P@10</th><th>TIPS cluster P@10</th></tr>",
    ]
    for _, row in merged.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['benchmark_type']))}</td>"
            f"<td>{html.escape(str(row['query']))}</td>"
            f"<td>{fmt(row.get('mrr_radseg'))}</td>"
            f"<td>{fmt(row.get('mrr_tips'))}</td>"
            f"<td>{fmt(row.get('recall@10_radseg'))}</td>"
            f"<td>{fmt(row.get('recall@10_tips'))}</td>"
            f"<td>{fmt(row.get('cluster_precision@10_radseg'))}</td>"
            f"<td>{fmt(row.get('cluster_precision@10_tips'))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def build_ground_truth_lookup(ground_truth):
    lookup = {}
    for item in ground_truth:
        lookup[(item["benchmark_type"], item["query"])] = item
    return lookup


def render_ground_truth_strip(gt_entry, images_root, max_items=4):
    if gt_entry is None:
        return "<div style='color:#777;'>No ground truth entry found.</div>"

    cards = []
    for image_id in gt_entry.get("relevant_images", [])[:max_items]:
        image_path = (images_root / image_id).resolve()
        if not image_path.exists():
            continue
        cards.append(
            "<div style='width:220px;'>"
            f"<img src='{html.escape(image_path.as_uri())}' style='width:100%;height:220px;object-fit:contain;border:1px solid #ddd;border-radius:8px;background:#fff;' />"
            f"<div style='font-size:12px;margin-top:6px;word-break:break-word;'>{html.escape(image_id)}</div>"
            "</div>"
        )

    if not cards:
        return "<div style='color:#777;'>No local ground truth images could be rendered.</div>"

    return (
        "<div style='margin-top:14px;'>"
        "<div style='font-weight:700;margin-bottom:8px;'>Ground truth samples</div>"
        "<div style='display:flex;gap:12px;flex-wrap:wrap;'>"
        + "".join(cards)
        + "</div></div>"
    )


def visualization_gallery(radseg_dir, tips_dir, image_a, image_b, ground_truth, images_root):
    merged = image_a[["benchmark_type", "query"]].merge(
        image_b[["benchmark_type", "query"]],
        on=["benchmark_type", "query"],
        how="inner",
    ).drop_duplicates().sort_values(["benchmark_type", "query"])
    gt_lookup = build_ground_truth_lookup(ground_truth)

    rows = [
        "<h2>Visualization Side-by-side</h2>",
        "<p>Each row shows the same query under the same benchmark settings, with RADSeg on the left and TIPS on the right. Ground truth sample images are rendered below for quick visual comparison.</p>",
    ]

    for _, row in merged.iterrows():
        benchmark_type = str(row["benchmark_type"])
        query = str(row["query"])
        query_slug = slugify_for_filename(query)
        radseg_img = radseg_dir / "visualizations" / benchmark_type / f"{query_slug}_heatmap.png"
        tips_img = tips_dir / "visualizations" / benchmark_type / f"{query_slug}_heatmap.png"
        if not radseg_img.exists() or not tips_img.exists():
            continue

        rows.append(
            "<div style='margin:28px 0 36px 0;border-top:1px solid #ddd;padding-top:20px;'>"
            f"<h3 style='margin:0 0 14px 0;'>{html.escape(benchmark_type)} — {html.escape(query)}</h3>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;'>"
            "<div>"
            "<div style='font-weight:700;margin-bottom:8px;'>RADSeg</div>"
            f"<img src='{html.escape(radseg_img.resolve().as_uri())}' style='width:100%;border:1px solid #ddd;border-radius:8px;' />"
            "</div>"
            "<div>"
            "<div style='font-weight:700;margin-bottom:8px;'>TIPS</div>"
            f"<img src='{html.escape(tips_img.resolve().as_uri())}' style='width:100%;border:1px solid #ddd;border-radius:8px;' />"
            "</div>"
            "</div>"
            f"{render_ground_truth_strip(gt_lookup.get((benchmark_type, query)), images_root)}"
            "</div>"
        )

    return "".join(rows)


def build_html(radseg_dir, tips_dir, out_path, images_root):
    radseg_dir, radseg_summary, radseg_image, radseg_cluster, radseg_gt = load_summary(radseg_dir)
    tips_dir, tips_summary, tips_image, tips_cluster, _ = load_summary(tips_dir)
    images_root = Path(images_root)

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>RADSeg vs TIPS Benchmark Comparison</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;max-width:1400px;margin:24px auto;padding:0 16px;line-height:1.45;}",
        "table{border-collapse:collapse;width:100%;margin:16px 0 32px 0;}",
        "th,td{border:1px solid #ddd;padding:8px 10px;text-align:left;vertical-align:top;}",
        "th{background:#f5f5f5;position:sticky;top:0;}",
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;}",
        ".card{border:1px solid #ddd;border-radius:10px;padding:16px;background:#fafafa;}",
        "code{background:#f0f0f0;padding:2px 6px;border-radius:4px;}",
        "</style></head><body>",
        "<h1>RADSeg vs TIPS Benchmark Comparison</h1>",
        "<p>This comparison uses the current landmark-heavy benchmark with <code>negative_text=background</code>. Cities are disabled; the query mix is concept + landmark only.</p>",
        "<div class='grid'>",
        "<div class='card'>"
        "<h2>RADSeg Report</h2>"
        f"<p><strong>Directory:</strong> <code>{html.escape(str(radseg_dir))}</code></p>"
        f"<p><a href='{html.escape(str((radseg_dir / 'query_browser_report.html').resolve().as_uri()))}'>Open RADSeg query browser HTML</a></p>"
        "</div>",
        "<div class='card'>"
        "<h2>TIPS Report</h2>"
        f"<p><strong>Directory:</strong> <code>{html.escape(str(tips_dir))}</code></p>"
        f"<p><a href='{html.escape(str((tips_dir / 'query_browser_report.html').resolve().as_uri()))}'>Open TIPS query browser HTML</a></p>"
        "</div>",
        "</div>",
        family_table(radseg_summary, tips_summary, "image_level_family_macro", "Image-level Family Macro Metrics"),
        family_table(radseg_summary, tips_summary, "cluster_level_family_macro", "Cluster-level Family Macro Metrics"),
        per_query_table(radseg_image, tips_image, radseg_cluster, tips_cluster),
        visualization_gallery(radseg_dir, tips_dir, radseg_image, tips_image, radseg_gt, images_root),
        "</body></html>",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(html_parts), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark report directories and render a side-by-side HTML.")
    parser.add_argument("--radseg_report", required=True)
    parser.add_argument("--tips_report", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--images_root", default="images")
    args = parser.parse_args()

    output_path = build_html(args.radseg_report, args.tips_report, args.output_html, args.images_root)
    print(f"Saved comparison HTML to {output_path}")


if __name__ == "__main__":
    main()
