"""
run_ingestion.py

CLI entry point for Phase 1: PDF -> typed segments.

Usage:
    python run_ingestion.py <pdf_path> [--out data/output/segments.json] [--no-images]

Set ANTHROPIC_API_KEY in your environment to enable image captioning.
Without it, images are still extracted but captions will be a placeholder.
"""

import argparse
import json
import os

from src.ingestion.segment_builder import build_segments


def main():
    parser = argparse.ArgumentParser(description="Run STRATA Phase 1 ingestion on a PDF.")
    parser.add_argument("pdf_path", help="Path to the input PDF")
    parser.add_argument("--out", default="data/output/segments.json", help="Where to write the segment JSON")
    parser.add_argument("--image-dir", default="data/output/extracted_images", help="Where to save extracted images")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction/captioning")
    parser.add_argument("--allow-text-table-fallback", action="store_true",
                         help="Enable experimental text-alignment table detection (see table_detector.py known limitations)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    segments = build_segments(
        args.pdf_path,
        image_output_dir=args.image_dir,
        caption_images=not args.no_images,
    )

    out_data = [
        {
            "page": s.page,
            "type": s.seg_type,
            "text": s.text,
            "bbox": s.bbox,
            "table_id": s.table_id,
        }
        for s in segments
    ]

    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)

    type_counts = {}
    for s in segments:
        type_counts[s.seg_type] = type_counts.get(s.seg_type, 0) + 1

    print(f"Ingested {args.pdf_path}")
    print(f"  {len(segments)} total segments -> {args.out}")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")


if __name__ == "__main__":
    main()
