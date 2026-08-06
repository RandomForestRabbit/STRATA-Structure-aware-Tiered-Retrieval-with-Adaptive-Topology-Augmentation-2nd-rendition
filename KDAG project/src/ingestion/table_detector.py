"""
table_detector.py

Detects and extracts tables from a PDF using pdfplumber. Tables are returned
as structured rows (list of cell values) rather than flattened text, along
with their bounding box on the page so the segment builder can exclude
overlapping raw text blocks and avoid duplicate content.

This is Phase 1, Step 2 of the STRATA pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional

import pdfplumber


@dataclass
class TableRow:
    page: int
    bbox: tuple             # bbox of the whole table this row belongs to
    row_index: int
    cells: List[Optional[str]]
    header: Optional[List[str]] = None

    @property
    def text(self) -> str:
        """Flatten the row into a single text string, preserving column pairing."""
        if self.header and len(self.header) == len(self.cells):
            parts = [
                f"{h.strip()}: {c.strip() if c else ''}"
                for h, c in zip(self.header, self.cells)
                if h
            ]
            return "; ".join(parts)
        return " | ".join(c.strip() if c else "" for c in self.cells)


DEFAULT_STRATEGIES = [
    # Works for tables with visible ruling lines
    {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
    # Fallback for borderless / whitespace-aligned tables (common in LaTeX
    # exports and many real-world PDFs with no visible gridlines)
    {"vertical_strategy": "text", "horizontal_strategy": "text"},
]


def _looks_like_header_cell(text: Optional[str]) -> bool:
    """A real header cell is short and label-like. Long, sentence-like text
    in a 'header' position is a strong sign the text-alignment strategy has
    misread a wrapped paragraph as a table (each wrapped line becomes a
    'row', and the first line becomes a bogus 'header')."""
    if not text:
        return True  # empty header cells are common and not suspicious on their own
    words = text.strip().split()
    return len(words) <= 6 and not text.strip().endswith((".", ",", ";"))


def _is_plausible_table(data: List[List[Optional[str]]]) -> bool:
    """
    Sanity check to reject false positives from the text-alignment detection
    strategy. Real tables have short, label-like header cells; misdetected
    paragraphs have long sentence-fragment 'headers' because the first
    wrapped line of the paragraph gets mistaken for a header row. This is a
    heuristic, not a guarantee -- always spot-check text-strategy results on
    your own corpus.
    """
    if len(data) < 2:
        return False

    header = data[0]
    non_empty_headers = [c for c in header if c and c.strip()]
    if not non_empty_headers:
        return False

    header_like_count = sum(1 for c in non_empty_headers if _looks_like_header_cell(c))
    return (header_like_count / len(non_empty_headers)) >= 0.75


def extract_tables(pdf_path: str, table_settings_list: Optional[List[dict]] = None,
                    allow_text_strategy_fallback: bool = False) -> List[TableRow]:
    """
    Extract every table on every page as a list of TableRow objects,
    one per row, each tagged with the parent table's bounding box so
    downstream chunking can treat 'inside this bbox' as a hard boundary.

    By default, only line-based detection is used (reliable, low false-positive
    rate). Set allow_text_strategy_fallback=True to additionally try
    whitespace/text-alignment-based detection for borderless tables (common
    in LaTeX-exported PDFs).

    KNOWN LIMITATION: the text-alignment strategy frequently misreads a
    wrapped prose paragraph as a multi-column table, because word-wrap
    whitespace looks like column gaps to a purely geometric detector. The
    _is_plausible_table() header-shape check catches some obvious cases but
    is NOT reliable -- it does not catch a paragraph whose wrapped-line
    fragments happen to be short (see project README, "Known Limitations").
    Treat allow_text_strategy_fallback=True as experimental: always spot-check
    its output before trusting it in an eval pipeline, and prefer running it
    only on pages you already suspect contain a borderless table.
    """
    rows: List[TableRow] = []
    strategies = table_settings_list or (DEFAULT_STRATEGIES if allow_text_strategy_fallback else DEFAULT_STRATEGIES[:1])

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            found_tables = []
            for settings in strategies:
                found_tables = page.find_tables(table_settings=settings)
                if found_tables:
                    break  # stop at the first strategy that finds anything

            for table in found_tables:
                data = table.extract()
                if not _is_plausible_table(data):
                    continue  # skip empty / single-row / misdetected-paragraph false positives

                header = data[0]
                bbox = table.bbox  # (x0, top, x1, bottom) in pdfplumber coords

                for i, raw_row in enumerate(data[1:], start=1):
                    rows.append(TableRow(
                        page=page_num,
                        bbox=bbox,
                        row_index=i,
                        cells=raw_row,
                        header=header,
                    ))

    return rows


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python table_detector.py <pdf_path>")
        sys.exit(1)

    rows = extract_tables(path)
    if not rows:
        print("No tables detected.")
    for r in rows[:15]:
        print(f"[p{r.page} row{r.row_index}] {r.text}")
    print(f"\n...{len(rows)} total table rows extracted")
