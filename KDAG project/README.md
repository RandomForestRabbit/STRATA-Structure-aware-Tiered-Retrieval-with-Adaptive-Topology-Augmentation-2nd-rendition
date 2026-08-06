# STRATA
### Structure-aware Tiered Retrieval with Adaptive Topology-Augmentation

STRATA is a retrieval-augmented generation (RAG) system for PDF question answering
that goes beyond fixed-size chunking and flat vector search. It combines two ideas:

1. **Hybrid Boundary-Aware Chunking** — chunk boundaries are determined by a
   combined signal of semantic drift (embedding similarity between adjacent
   segments), entity/topic continuity (noun-phrase overlap), and document
   structure (headings, table rows, list items) — instead of splitting on a
   fixed token count.

2. **Graph-Augmented Retrieval** — chunks are linked into a lightweight
   entity-relation graph (LightRAG-style dual-level indexing: fine-grained
   entity links + high-level topic clusters), enabling multi-hop retrieval
   that plain vector similarity search misses — e.g., pulling in a chunk that
   shares an entity with the query but doesn't read as topically similar on
   its own.

STRATA is evaluated across four structurally distinct PDF types — tabular
documents, image-heavy documents, long-form prose, and short instructional
text — to measure where structure-aware chunking and graph augmentation
actually help, and where they don't.

## Why

Standard RAG pipelines chunk documents by token count and retrieve purely by
embedding similarity. This breaks down in predictable ways: chunking mid-table-row
severs an answer from its context, and vector search alone can miss relevant
facts that are lexically distant from the query even when they're directly
connected via a shared entity. STRATA targets both failure modes directly and
measures the improvement — and the limits — on a per-document-type basis.

**Scope note:** this is an applied systems/engineering project, not a claim
of novel algorithmic research. Its components (semantic chunking, structural
chunking, LightRAG-style graph retrieval) are individually known techniques;
the contribution here is combining them with a shared continuity signal and
rigorously measuring where the combination helps versus where it doesn't
across document types.

## Pipeline

```
PDF → Typed Segment Extraction (text / table / image caption / heading)   [Phase 1 — this release]
    → Hybrid Boundary-Aware Chunker                                       [Phase 2]
    → Entity/Relation Extraction → Dual-Level Graph (LightRAG-style)      [Phase 3]
    → Query-Time Retrieval (vector search + 1–2 hop graph expansion)      [Phase 4]
    → LLM Generation                                                      [Phase 5]
    → Evaluation across document types (Recall@k, MRR, answer correctness) [Phase 6]
```

## Status

🚧 **Phase 1 (PDF ingestion) implemented.** Chunking, graph construction, and
retrieval are in progress. See `Roadmap` below.

## Phase 1: PDF Ingestion

Converts a raw PDF into an ordered stream of **typed segments** — the shared
input format every later phase builds on.

| Module | Responsibility |
|---|---|
| `src/ingestion/layout_extractor.py` | Extracts text blocks with font size / position via PyMuPDF; classifies each as `heading`, `prose`, or `list_item` based on document-relative font-size statistics. |
| `src/ingestion/table_detector.py` | Detects tables via pdfplumber and extracts them as structured rows (not flattened text), preserving header-to-cell pairing. |
| `src/ingestion/image_captioner.py` | Extracts embedded images and generates short text captions via a vision-capable LLM, so images become retrievable through the same text pipeline. |
| `src/ingestion/segment_builder.py` | Merges the above into one ordered, deduplicated segment stream in reading order, with table/image regions excluded from raw text blocks to avoid double-counting content. |

### Usage

```bash
pip install -r requirements.txt

# Set this to enable image captioning (optional — ingestion works without it,
# images are still extracted, captions are just left as placeholders)
export ANTHROPIC_API_KEY=your_key_here

python run_ingestion.py path/to/document.pdf --out data/output/segments.json
```

Each module is also runnable standalone for debugging:
```bash
python -m src.ingestion.layout_extractor path/to/document.pdf
python -m src.ingestion.table_detector path/to/document.pdf
python -m src.ingestion.image_captioner path/to/document.pdf ./extracted_images
python -m src.ingestion.segment_builder path/to/document.pdf
```

### Known Limitations

- **Table detection on borderless tables is unreliable.** The default
  detector uses ruling-line detection (`vertical_strategy: lines`), which
  misses tables with no visible gridlines (common in LaTeX-exported PDFs).
  An experimental whitespace/text-alignment fallback
  (`allow_text_strategy_fallback=True`) exists but **frequently misreads
  wrapped prose paragraphs as tables** — a header-shape heuristic filters
  some false positives but does not catch all of them. Treat this fallback
  as unverified and spot-check its output before trusting it in an eval
  pipeline. This is an open problem, not a solved one — a more robust fix
  (e.g., column-alignment consistency scoring across rows, or a
  vision-based table detector) is a natural next step.
- **Image bounding boxes** are best-effort (`page.get_image_rects`) and can
  be `None` for some embedding methods; the segment builder falls back to
  `y0=0.0` for ordering in that case, which may place the caption out of
  true reading order.
- **Heading detection** is a font-size heuristic (15% above document-median
  body text, or bold), which works well for most academic/report-style PDFs
  but can misfire on documents with unconventional typography.

## Roadmap

- [x] Phase 1 — Typed segment extraction (text, tables, images)
- [ ] Phase 2 — Hybrid boundary-aware chunker (drift + continuity + structure)
- [ ] Phase 3 — Entity/relation extraction + dual-level graph construction
- [ ] Phase 4 — Query-time retrieval with graph-hop expansion
- [ ] Phase 5 — Generation
- [ ] Phase 6 — Cross-document-type evaluation harness (tabular / image-heavy / long-form / instructional)

## Repo Structure

```
strata/
├── src/
│   └── ingestion/
│       ├── layout_extractor.py
│       ├── table_detector.py
│       ├── image_captioner.py
│       └── segment_builder.py
├── tests/
│   └── test_ingestion.py
├── data/
│   ├── sample_pdfs/
│   └── output/
├── run_ingestion.py
├── requirements.txt
└── README.md
```
