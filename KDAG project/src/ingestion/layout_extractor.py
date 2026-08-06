"""
layout_extractor.py

Extracts text blocks from a PDF with layout metadata (font size, bounding box,
page number) using PyMuPDF, and classifies each block as a heading or body
paragraph based on font-size statistics for the document.

This is Phase 1, Step 1 of the STRATA pipeline: typed segment extraction.
"""

from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional

import fitz  # PyMuPDF


@dataclass
class TextBlock:
    page: int
    bbox: tuple            # (x0, y0, x1, y1)
    text: str
    avg_font_size: float
    is_bold: bool
    block_type: str = "prose"   # "heading" | "prose" | "list_item"


def _looks_like_list_item(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    list_markers = ("- ", "* ", "• ", "◦ ")
    if stripped.startswith(list_markers):
        return True
    # numbered list: "1.", "1)", "(1)"
    head = stripped.split(" ", 1)[0].rstrip(".)")
    if head.isdigit():
        return True
    return False


def extract_text_blocks(pdf_path: str) -> List[TextBlock]:
    """
    Extract raw text blocks with font-size and position metadata.
    Classifies each block as heading / prose / list_item based on
    document-relative font-size statistics.
    """
    doc = fitz.open(pdf_path)
    raw_blocks = []
    font_sizes = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block, 1 = image block
                continue
            block_text_parts = []
            sizes = []
            bold_flags = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if not span_text.strip():
                        continue
                    block_text_parts.append(span_text)
                    sizes.append(span.get("size", 0.0))
                    bold_flags.append(bool(span.get("flags", 0) & 2 ** 4))

            if not block_text_parts:
                continue

            text = " ".join(block_text_parts).strip()
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0
            is_bold = sum(bold_flags) > len(bold_flags) / 2 if bold_flags else False

            raw_blocks.append({
                "page": page_num,
                "bbox": tuple(block["bbox"]),
                "text": text,
                "avg_font_size": avg_size,
                "is_bold": is_bold,
            })
            font_sizes.append(avg_size)

    doc.close()

    if not font_sizes:
        return []

    body_font_size = median(font_sizes)
    heading_threshold = body_font_size * 1.15  # >15% larger than median body text

    blocks: List[TextBlock] = []
    for rb in raw_blocks:
        if rb["avg_font_size"] >= heading_threshold or (rb["is_bold"] and rb["avg_font_size"] > body_font_size):
            block_type = "heading"
        elif _looks_like_list_item(rb["text"]):
            block_type = "list_item"
        else:
            block_type = "prose"

        blocks.append(TextBlock(
            page=rb["page"],
            bbox=rb["bbox"],
            text=rb["text"],
            avg_font_size=rb["avg_font_size"],
            is_bold=rb["is_bold"],
            block_type=block_type,
        ))

    return blocks


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python layout_extractor.py <pdf_path>")
        sys.exit(1)

    blocks = extract_text_blocks(path)
    for b in blocks[:20]:
        print(f"[p{b.page}] ({b.block_type}, size={b.avg_font_size:.1f}) {b.text[:80]}")
    print(f"\n...{len(blocks)} total blocks extracted")
