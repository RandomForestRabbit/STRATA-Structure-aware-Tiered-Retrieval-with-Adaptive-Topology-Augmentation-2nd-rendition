"""
image_captioner.py

Extracts embedded images from a PDF and generates short text captions using
a local LLaVA vision-language model. Captions act as a text proxy for the
image so it can be retrieved through the same vector/graph pipeline as
regular text chunks, rather than requiring a separate image-embedding index.

This is Phase 1, Step 3 of the STRATA pipeline.

Requires `torch` and `transformers` to be installed, plus enough GPU/CPU
memory to hold the LLaVA weights. The model is loaded once (lazily, on first
use) and reused for every image rather than being reloaded per call. Without
the required packages / hardware, images are still extracted and saved, but
captions are left as a placeholder so the rest of the pipeline can still be
exercised.

Interface is unchanged from the API-backed versions of this file:
extract_and_caption_images(pdf_path, output_dir) -> List[ImageSegment]
"""

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

# Any LLaVA-family checkpoint on the HF hub works here; swap as needed.
LLAVA_MODEL_ID = os.environ.get("LLAVA_MODEL_ID", "llava-hf/llava-interleave-qwen-0.5b-hf")

# Module-level cache so the (large) model/processor are loaded only once,
# no matter how many images get captioned across the run.
_model = None
_processor = None
_device = None


def _load_llava():
    """Lazily load and cache the LLaVA model + processor."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    import torch
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    if torch.cuda.is_available():
        _device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        _device = "mps"
        dtype = torch.float16
    else:
        _device = "cpu"
        dtype = torch.float32

    _processor = LlavaProcessor.from_pretrained(LLAVA_MODEL_ID)
    _model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(_device)
    _model.eval()

    return _model, _processor, _device


def _caption_with_llava(image_path: str, media_type: str = "image/png") -> str:
    try:
        import torch
        from PIL import Image
    except ImportError:
        return "[caption unavailable: torch/transformers/Pillow not installed]"

    try:
        model, processor, device = _load_llava()
    except Exception as e:
        return f"[caption unavailable: failed to load LLaVA model ({str(e)})]"

    try:
        image = Image.open(image_path).convert("RGB")

        # LLaVA chat template expects an explicit <image> placeholder token
        # embedded in the prompt text, unlike the API-based messages format.
        prompt = f"USER: <image>\n{CAPTION_PROMPT}\nASSISTANT:"

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        decoded = processor.decode(output_ids[0], skip_special_tokens=True)

        # The model echoes the prompt before the answer; strip everything
        # up to and including "ASSISTANT:" to isolate just the caption.
        if "ASSISTANT:" in decoded:
            caption = decoded.split("ASSISTANT:", 1)[1].strip()
        else:
            caption = decoded.strip()

        return caption or "[empty caption returned]"
    except Exception as e:
        return f"[error generating caption: {str(e)}]"


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
            caption = _caption_with_llava(out_path, media_type=media_type)

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
