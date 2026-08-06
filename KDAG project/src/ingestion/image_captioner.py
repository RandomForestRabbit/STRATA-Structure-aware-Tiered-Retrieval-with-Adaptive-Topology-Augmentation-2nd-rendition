"""
image_captioner.py

Extracts embedded images from a PDF and generates short text captions using
a vision-capable LLM. Captions act as a text proxy for the image so it can
be retrieved through the same vector/graph pipeline as regular text chunks,
rather than requiring a separate image-embedding index.

This is Phase 1, Step 3 of the STRATA pipeline.

Requires ANTHROPIC_API_KEY to be set in the environment to actually generate
captions. Without it, images are still extracted and saved, but captions are
left as a placeholder so the rest of the pipeline can still be exercised.
"""

import base64
import os
from dataclasses import dataclass
from typing import List, Optional

import fitz  # PyMuPDF


@dataclass
class ImageSegment:
    page: int
    bbox: Optional[tuple]
    image_path: str
    caption: str


CAPTION_PROMPT = (
    "Describe this image in 1-2 concise sentences, focused on any facts, "
    "labels, numbers, or named entities visible in it. This description will "
    "be used as a searchable text stand-in for the image in a retrieval "
    "system, so prioritize concrete, specific content over general visual "
    "description (e.g. prefer 'bar chart showing Q3 revenue at $4.2M' over "
    "'a colorful chart')."
)


def _caption_with_claude(image_path: str, media_type: str = "image/png") -> str:
    try:
        import anthropic
    except ImportError:
        return "[caption unavailable: anthropic SDK not installed]"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[caption unavailable: ANTHROPIC_API_KEY not set]"

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }],
    )
    text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    return " ".join(text_parts).strip() or "[empty caption returned]"


def extract_and_caption_images(pdf_path: str, output_dir: str) -> List[ImageSegment]:
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    segments: List[ImageSegment] = []

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            filename = f"page{page_num}_img{img_index}.{ext}"
            out_path = os.path.join(output_dir, filename)
            with open(out_path, "wb") as f:
                f.write(image_bytes)

            # Best-effort bbox lookup (not guaranteed for every embedding method)
            bbox = None
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    bbox = tuple(rects[0])
            except Exception:
                pass

            media_type = f"image/{ext if ext in ('png', 'jpeg', 'jpg') else 'png'}"
            caption = _caption_with_claude(out_path, media_type=media_type)

            segments.append(ImageSegment(
                page=page_num,
                bbox=bbox,
                image_path=out_path,
                caption=caption,
            ))

    doc.close()
    return segments


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./extracted_images"
    if not path:
        print("Usage: python image_captioner.py <pdf_path> [output_dir]")
        sys.exit(1)

    segs = extract_and_caption_images(path, out_dir)
    if not segs:
        print("No images found.")
    for s in segs:
        print(f"[p{s.page}] {s.image_path} -> {s.caption}")
    print(f"\n...{len(segs)} images extracted to {out_dir}")
