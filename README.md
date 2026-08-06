
### Structure-aware Tiered Retrieval with Adaptive Topology-Augmentation

STRATA is a retrieval-augmented generation (RAG) system for PDF question answering
that goes beyond fixed-size chunking and flat vector search. It combines three ideas:

1. **Hybrid Boundary-Aware Chunking** — chunk boundaries are determined by a
   combined signal of semantic drift (embedding similarity between adjacent
   segments), entity continuity (named-entity overlap via NER), and document
   structure (headings, table rows, list items, images) — instead of splitting
   on a fixed token count.

2. **Memory-Augmented Graph Retrieval** — chunks are linked into an
   entity-relation graph built against a persistent, three-tier **Global
   Memory** (inspired by MemGraphRAG, KDD 2026), which reconciles entities and
   facts *across* documents — not just within one — so the same real-world
   entity mentioned in different PDFs resolves to one graph node instead of
   several disconnected ones, and contradictory facts get flagged rather than
   silently overwritten.

3. **Fused Retrieval** — query time combines standard vector search with
   1–2 hop graph traversal, merges and reranks both result sets, and
   explicitly labels context by source (direct match / graph-hop / contested
   fact) so both the LLM and later evaluation can tell which retrieval path
   actually contributed to a correct answer.

STRATA is evaluated across four structurally distinct PDF types — tabular
documents, image-heavy documents, long-form prose, and short instructional
text — to measure where structure-aware chunking and graph augmentation
actually help, and where they don't.

## Why

Standard RAG pipelines chunk documents by token count and retrieve purely by
embedding similarity. This breaks down in predictable, well-known ways:
chunking mid-table-row severs an answer from its context, vector search alone
can miss relevant facts that are lexically distant from the query even when
they're directly connected via a shared entity, and per-chunk graph
extraction (as in plain LightRAG) produces fragmented, inconsistent entities
across a multi-document corpus. STRATA targets all three failure modes
directly and measures the improvement — and the limits — on a
per-document-type basis.

**Scope note:** this is an applied systems/engineering project, not a claim
of novel algorithmic research. Its components (semantic chunking, structural
chunking, memory-augmented graph retrieval) are individually known
techniques, simplified from their original published forms to fit this
project's scope — see `docs/phase3_memory_graph_design.md` for an explicit
account of what was simplified and why. The contribution here is combining
these pieces thoughtfully and rigorously measuring where the combination
helps versus where it doesn't, across document types — not inventing new
algorithms.

## Pipeline

```
PDF(s) → Typed Segment Extraction (text / table / image caption / heading)     [Phase 1 — done]
       → Hybrid Boundary-Aware Chunking (drift + entity continuity + structure) [Phase 2a — in progress]
       → Embed chunks → Vector Store (Chroma)                                   [Phase 2b — pending]
       → Entity/Relation Extraction → Global Memory → Graph (MemGraphRAG-style) [Phase 3 — designed]
       → Query-Time Retrieval: vector search + graph-hop expansion + fusion     [Phase 4 — designed]
       → LLM Generation with sourced, labeled context                          [Phase 5 — designed]
       → Evaluation across document types (Recall@k, MRR, correctness, latency) [Phase 6 — not started]
```

## Status

🚧 **Phase 1 complete and tested.** **Phase 2 in progress** — shared embedding
and entity-extraction utilities are built; the chunker itself and the vector
store step are next. Phases 3–5 are fully designed (see `docs/`) but not yet
implemented. Phase 6 has not started.

---

## Phase 1: PDF Ingestion ✅

Converts a raw PDF into an ordered stream of **typed segments** — the shared
input format every later phase builds on.

| Module | Responsibility |
|---|---|
| `src/ingestion/layout_extractor.py` | Extracts text blocks with font size / position via PyMuPDF; classifies each as `heading`, `prose`, or `list_item` based on document-relative font-size statistics. |
| `src/ingestion/table_detector.py` | Detects tables via pdfplumber and extracts them as structured rows (not flattened text), preserving header-to-cell pairing. Line-based detection by default; an experimental text-alignment fallback exists but is off by default (see Known Limitations). |
| `src/ingestion/image_captioner.py` | Extracts embedded images and generates short text captions via a vision-capable LLM, so images become retrievable through the same text pipeline. |
| `src/ingestion/segment_builder.py` | Merges the above into one ordered, deduplicated segment stream in reading order, with table/image regions excluded from raw text blocks to avoid double-counting content. |

### Usage

```bash
pip install -r requirements.txt

# Set this to enable image captioning (optional — ingestion works without it,
# images are still extracted, captions are just left as placeholders)
export API_KEY = your_key_here

python run_ingestion.py path/to/document.pdf --out data/output/segments.json
```

Each module is also runnable standalone for debugging (run as a module, from
the project root, due to relative imports):
```bash
python -m src.ingestion.layout_extractor path/to/document.pdf
python -m src.ingestion.table_detector path/to/document.pdf
python -m src.ingestion.image_captioner path/to/document.pdf ./extracted_images
python -m src.ingestion.segment_builder path/to/document.pdf
```

### Known Limitations

- **Table detection on borderless tables is unreliable.** The default
  detector uses ruling-line detection, which misses tables with no visible
  gridlines (common in LaTeX-exported PDFs). An experimental text-alignment
  fallback (`allow_text_strategy_fallback=True`) exists but **frequently
  misreads wrapped prose paragraphs as tables** — a header-shape heuristic
  filters some false positives but not all. Treat this fallback as
  unverified; a more robust fix (column-alignment consistency scoring, or a
  vision-based table detector) is a natural next step, not yet built.
- **Image bounding boxes** are best-effort and can be `None` for some
  embedding methods; ordering falls back to `y0=0.0` in that case, which may
  place the caption out of true reading order.
- **Heading detection** is a font-size heuristic (15% above document-median
  body text, or bold) — works for most academic/report-style PDFs, can
  misfire on documents with unconventional typography.

---

## Phase 2: Chunking & Embedding 🚧 in progress

Turns Phase 1's segment stream into retrieval-ready chunks with stored
embeddings.

| Module | Responsibility | Status |
|---|---|---|
| `src/embeddings/embedder.py` | Shared wrapper around a Hugging Face sentence-embedding model (default `all-MiniLM-L6-v2`); used for chunker drift-scoring, chunk storage, and later query embedding — one model, reused everywhere for consistency. Tracks each model's max-sequence-length limit (MiniLM: 256 tokens) so chunk size can be validated against it. | ✅ built |
| `src/chunking/entity_extractor.py` | spaCy NER wrapper for the continuity signal; restricted to PERSON/ORG/GPE/PRODUCT/EVENT/etc. (excludes noisy categories like CARDINAL). Same extraction logic is reused as-is for Phase 3's entity/relation extraction. | ✅ built |
| `src/chunking/hybrid_chunker.py` | Combines: **hard rules** (never split inside a table, force-split on headings, keep image captions atomic) + **soft scoring** (embedding drift + entity-continuity, weighted and thresholded) + **min/max token caps** as a safety net. | 🔜 next |
| `src/chunking/vector_store.py` | Embeds final chunks and stores them (with `doc_id`, page, source-segment-type metadata) in a Chroma collection. | 🔜 pending |

### Design notes worth knowing before reading the code

- **Granularity decision:** chunking operates on Phase 1's segments *as-is*
  (paragraph/row/caption-level) — segments are **not** further split into
  individual sentences before scoring. This is a deliberate simplification:
  cheaper, and reasonable given these documents are structured
  (headings/lists/tables) rather than dense uninterrupted prose. The
  trade-off: the chunker can only split *between* segments, never *inside*
  one, so a topic shift partway through one long paragraph won't be caught.
- **Continuity uses NER, not generic noun-phrase overlap.** This is narrower
  (misses topic continuity that doesn't involve a named entity) but keeps the
  signal consistent with Phase 3, which also needs named entities for the
  graph — one extraction pass' logic, reused for two purposes.
- **`doc_id` is threaded through every `Segment`/`Chunk`** from Phase 1
  onward, in preparation for Phase 3's cross-document memory — this is what
  eventually lets facts from different PDFs be traced back to their source.
- **Embedding-model token limits matter for chunk size.** MiniLM truncates
  anything past 256 tokens silently. If `hybrid_chunker.py`'s `max_tokens` is
  set above the active embedding model's limit, longer chunks will be
  embedded incompletely — worth checking `embedder.max_tokens` against your
  chunking config before trusting retrieval results.

### Usage (currently available modules)

```bash
python -m src.embeddings.embedder            # sanity check: drift score on a related vs unrelated pair
python -m src.chunking.entity_extractor       # sanity check: entity extraction + continuity score
```

---

## Phase 3: Memory-Augmented Graph Construction 📐 designed, not built

Full design locked in **`docs/phase3_memory_graph_design.md`** — read that
file for the complete spec. Summary:

- A persistent **three-tier Global Memory** shared across the *entire*
  corpus (not per-document): an **Ontology Layer** (entity/relation type
  registry), a **Fact Layer** (reconciled, deduplicated facts with a
  `confirmed`/`contested`/`superseded` status), and a **Passage Layer**
  (evidence links back to source chunks).
- **Entity resolution**: exact match → embedding similarity → LLM
  confirmation (only for ambiguous cases) — this is what makes "Priya Shah"
  in one PDF and "P. Shah" in another resolve to one graph node.
- **Conflict detection**: when two chunks make contradictory claims about the
  same entity pair, a single LLM adjudication call resolves it if possible,
  or flags it `contested` if genuinely ambiguous — surfaced to the LLM at
  generation time rather than silently picked.
- Explicitly a **simplified, single-agent adaptation** of MemGraphRAG's
  multi-agent architecture — the design doc has a full table of what was
  deliberately not carried over, and why.

---

## Phase 4: Fused Retrieval 📐 designed, not built

- Query gets embedded *and* run through the same NER extractor to find seed
  entities for graph traversal.
- Vector search (Chroma) and graph-hop expansion (1–2 hops, capped) run
  independently, then get **fused**: deduplicated by chunk id, scored with a
  weighted combination (vector similarity + hop-distance decay + a bonus for
  chunks found by both), filtered by a minimum relevance floor, and packed
  into the LLM's context budget by token count (not a fixed chunk count).
- Optional cross-encoder **reranking** pass over the fused candidate pool —
  planned as a configurable ablation axis (Config D vs. Config E), not a
  silent default, so its actual latency/accuracy trade-off gets measured
  rather than assumed.

## Phase 5: Generation 📐 designed, not built

- Prompt assembly explicitly **labels** each piece of context by source
  (direct vector match / N-hop graph match / contested fact with both
  conflicting sources shown).
- Post-processing checks: citation-to-source verification, whether contested
  facts were correctly flagged (vs. silently resolved) by the LLM, and a
  hard fallback to "insufficient information" if fused retrieval returns
  nothing above the relevance floor — rather than letting the LLM answer
  from its own unrelated training knowledge.

## Phase 6: Evaluation ⏳ not started

Planned config matrix (fixed-token baseline / hybrid-chunking-only /
graph-only / full system / full system + rerank) run across 4 structurally
distinct document types, measuring Recall@k, MRR, answer correctness
(exact-match or LLM-judge), latency, cost, and — specifically — cross-PDF
query accuracy (questions whose answer requires linking facts from two
different documents), since that's the metric that directly validates
whether Phase 3's memory design achieved its actual goal.

---

## Repo Structure

```
strata/
├── src/
│   ├── ingestion/
│   │   ├── layout_extractor.py
│   │   ├── table_detector.py
│   │   ├── image_captioner.py
│   │   └── segment_builder.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── chunking/
│   │   ├── entity_extractor.py
│   │   ├── hybrid_chunker.py        [pending]
│   │   └── vector_store.py          [pending]
│   ├── graph/                        [pending — Phase 3]
│   └── retrieval/                    [pending — Phase 4]
├── docs/
│   └── phase3_memory_graph_design.md
├── tests/
│   └── test_ingestion.py
├── data/
│   ├── sample_pdfs/
│   └── output/
├── run_ingestion.py
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Phase 1 — Typed segment extraction (text, tables, images)
- [ ] Phase 2 — Hybrid chunking (drift + continuity + structure) + vector storage — *embedder & entity extractor done, chunker + vector store pending*
- [ ] Phase 3 — Memory-augmented graph construction (MemGraphRAG-inspired) — *design locked, code pending*
- [ ] Phase 4 — Fused retrieval (vector + graph-hop + rerank) — *design locked, code pending*
- [ ] Phase 5 — Generation with sourced, labeled context — *design locked, code pending*
- [ ] Phase 6 — Cross-document-type evaluation harness
