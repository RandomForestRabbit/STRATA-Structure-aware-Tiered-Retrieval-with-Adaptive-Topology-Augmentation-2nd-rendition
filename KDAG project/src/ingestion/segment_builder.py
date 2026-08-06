"""
segment_builder.py

Merges the outputs of layout_extractor, table_detector, and image_captioner
into a single, ordered stream of typed Segments per document. This unified
stream is the input to the hybrid chunker (Phase 2).

Overlap handling: raw text blocks that fall inside a detected table's
bounding box are dropped, since that region is already represented by
structured TableRow segments (prevents duplicating a table's text twice —
once as flattened prose, once as structured rows).

This is Phase 1, Step 4 of the STRATA pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional

from .layout_extractor import extract_text_blocks
from .table_detector import extract_tables
from .image_captioner import extract_and_caption_images


@dataclass
class Segment:
    page: int
    y0: float                 # vertical position, used for reading-order sort
    seg_type: str              # "heading" | "prose" | "list_item" | "table_row" | "image_caption"
    text: str
    bbox: Optional[tuple] = None
    table_id: Optional[int] = None   # groups rows belonging to the same table


def _boxes_overlap(a: tuple, b: tuple, threshold: float = 0.5) -> bool:
    """Return True if box `a` overlaps box `b` by more than `threshold` of a's area."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)

    if ix1 <= ix0 or iy1 <= iy0:
        return False

    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    return (intersection / area_a) >= threshold


def build_segments(pdf_path: str, image_output_dir: str = "./extracted_images",
                    caption_images: bool = True) -> List[Segment]:
    text_blocks = extract_text_blocks(pdf_path)
    table_rows = extract_tables(pdf_path)
    image_segments = extract_and_caption_images(pdf_path, image_output_dir) if caption_images else []

    # Build lookup of table bboxes per page, deduplicated, with a stable table_id
    table_boxes_per_page = {}
    table_id_by_bbox = {}
    next_table_id = 0
    for row in table_rows:
        key = (row.page, row.bbox)
        if key not in table_id_by_bbox:
            table_id_by_bbox[key] = next_table_id
            next_table_id += 1
        table_boxes_per_page.setdefault(row.page, set()).add(row.bbox)

    image_boxes_per_page = {}
    for img in image_segments:
        if img.bbox:
            image_boxes_per_page.setdefault(img.page, []).append(img.bbox)

    segments: List[Segment] = []

    # 1. Text blocks, excluding anything overlapping a detected table or image region
    for block in text_blocks:
        table_boxes = table_boxes_per_page.get(block.page, set())
        if any(_boxes_overlap(block.bbox, tb) for tb in table_boxes):
            continue

        image_boxes = image_boxes_per_page.get(block.page, [])
        if any(_boxes_overlap(block.bbox, ib) for ib in image_boxes):
            continue

        segments.append(Segment(
            page=block.page,
            y0=block.bbox[1],
            seg_type=block.block_type,
            text=block.text,
            bbox=block.bbox,
        ))

    # 2. Table rows, tagged with a shared table_id so the chunker can treat
    #    all rows of one table as belonging to the same structural unit
    for row in table_rows:
        tid = table_id_by_bbox[(row.page, row.bbox)]
        segments.append(Segment(
            page=row.page,
            y0=row.bbox[1],   # sort by table position; row_index breaks ties below
            seg_type="table_row",
            text=row.text,
            bbox=row.bbox,
            table_id=tid,
        ))

    # 3. Image captions
    for img in image_segments:
        segments.append(Segment(
            page=img.page,
            y0=img.bbox[1] if img.bbox else 0.0,
            seg_type="image_caption",
            text=f"[Image] {img.caption}",
            bbox=img.bbox,
        ))

    # Reading order: page, then vertical position. Table rows on the same
    # bbox are further ordered by their natural extraction order via a
    # stable sort (Python's sort is stable, so original relative order for
    # ties is preserved).
    segments.sort(key=lambda s: (s.page, s.y0))

    return segments


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python segment_builder.py <pdf_path>")
        sys.exit(1)

    segs = build_segments(path, caption_images=False)  # skip captioning for a quick smoke test
    for s in segs[:30]:
        print(f"[p{s.page} {s.seg_type}] {s.text[:80]}")
    print(f"\n...{len(segs)} total segments built")
