import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    logging.warning("sentence-transformers is not installed. Please install it.")

try:
    import spacy
except ImportError:
    logging.warning("spacy is not installed. Please install it.")

try:
    import tiktoken
except ImportError:
    logging.warning("tiktoken is not installed. Please install it.")

from src.ingestion.segment_builder import Segment

@dataclass
class Chunk:
    chunk_id: str           # unique ID (e.g. '{doc_id}_chunk_{n}')
    doc_id: str             # source document identifier
    text: str               # concatenated text of all segments in this chunk
    segments: List[dict]    # source segment references [{page, seg_type, bbox, text_preview}]
    page_start: int
    page_end: int
    chunk_type: str         # 'prose' | 'table' | 'image' | 'heading' | 'mixed'
    metadata: dict          # arbitrary metadata dict


class HybridChunker:
    def __init__(self, model_name='all-MiniLM-L6-v2', max_chunk_tokens=384, 
                 similarity_threshold=0.65, entity_weight=0.3, 
                 similarity_weight=0.7):
        self.max_chunk_tokens = max_chunk_tokens
        self.similarity_threshold = similarity_threshold
        self.entity_weight = entity_weight
        self.similarity_weight = similarity_weight
        
        try:
            self.embedder = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence-transformers model {model_name}: {e}")
            
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            raise RuntimeError(f"Failed to load tiktoken encoding: {e}")

    def _get_noun_phrases(self, text: str) -> set:
        doc = self.nlp(text)
        return set([chunk.text.lower() for chunk in doc.noun_chunks])

    def chunk_segments(self, segments: List[Segment], doc_id: str) -> List[Chunk]:
        if not segments:
            return []

        # 1. Structural grouping
        groups = []
        current_group = []
        current_heading_group = False

        for seg in segments:
            if seg.seg_type == 'table_row':
                # Tables are grouped separately later, but for now we put them in groups if they share table_id
                # Actually, let's just group consecutive table_rows with the same table_id
                if not current_group:
                    current_group.append(seg)
                elif current_group[-1].seg_type == 'table_row' and current_group[-1].table_id == seg.table_id:
                    current_group.append(seg)
                else:
                    groups.append(current_group)
                    current_group = [seg]
                current_heading_group = False
            elif seg.seg_type == 'image_caption':
                if current_group:
                    groups.append(current_group)
                groups.append([seg])
                current_group = []
                current_heading_group = False
            elif seg.seg_type == 'heading':
                if current_group:
                    groups.append(current_group)
                current_group = [seg]
                current_heading_group = True
            else: # prose or list_item
                current_group.append(seg)
        
        if current_group:
            groups.append(current_group)

        # 2 & 3: Soft boundary detection and Token budget packing
        final_chunks = []
        chunk_idx = 0

        for group in groups:
            if not group:
                continue
                
            group_type = group[0].seg_type
            
            # Atomic groups: Tables, Image captions, Heading groups
            is_atomic = False
            if group_type == 'table_row':
                is_atomic = True
            elif group_type == 'image_caption':
                is_atomic = True
            
            if is_atomic:
                chunk = self._create_chunk(group, doc_id, chunk_idx)
                final_chunks.append(chunk)
                chunk_idx += 1
                continue

            # For prose streams (including heading + following prose), apply soft boundaries and token packing
            sub_chunks = []
            current_sub_chunk = []
            current_tokens = 0
            
            # Embed all segments in the group for boundary detection
            texts = [s.text for s in group]
            embeddings = self.embedder.encode(texts, convert_to_tensor=True)
            noun_phrases = [self._get_noun_phrases(t) for t in texts]

            for i, seg in enumerate(group):
                seg_tokens = len(self.tokenizer.encode(seg.text))
                
                # If a single segment is too big, just force it in
                if not current_sub_chunk:
                    current_sub_chunk.append(seg)
                    current_tokens = seg_tokens
                    continue
                
                # Check soft boundary if it's not the first segment
                boundary_score = 0.0
                if i > 0:
                    sim = cos_sim(embeddings[i-1], embeddings[i]).item()
                    np_prev = noun_phrases[i-1]
                    np_curr = noun_phrases[i]
                    overlap = len(np_prev.intersection(np_curr)) / max(1, len(np_prev.union(np_curr)))
                    
                    boundary_score = self.similarity_weight * (1 - sim) + self.entity_weight * (1 - overlap)

                is_boundary = boundary_score > (1 - self.similarity_threshold)
                is_full = (current_tokens + seg_tokens) > self.max_chunk_tokens
                
                # If we hit a boundary OR the token budget, we wrap up the current chunk
                # UNLESS it's a heading group and we're at the very first element (heading itself), 
                # we don't want to strand the heading alone if possible, but budget rules still apply.
                if is_boundary or is_full:
                    final_chunks.append(self._create_chunk(current_sub_chunk, doc_id, chunk_idx))
                    chunk_idx += 1
                    current_sub_chunk = [seg]
                    current_tokens = seg_tokens
                else:
                    current_sub_chunk.append(seg)
                    current_tokens += seg_tokens
                    
            if current_sub_chunk:
                final_chunks.append(self._create_chunk(current_sub_chunk, doc_id, chunk_idx))
                chunk_idx += 1

        return final_chunks

    def _create_chunk(self, segments: List[Segment], doc_id: str, chunk_idx: int) -> Chunk:
        text = "\n".join(s.text for s in segments)
        page_start = min(s.page for s in segments)
        page_end = max(s.page for s in segments)
        
        # Determine chunk type
        types = set(s.seg_type for s in segments)
        if len(types) == 1:
            if 'table_row' in types:
                chunk_type = 'table'
            elif 'image_caption' in types:
                chunk_type = 'image'
            elif 'heading' in types:
                chunk_type = 'heading'
            else:
                chunk_type = 'prose'
        else:
            chunk_type = 'mixed'
            
        segment_refs = []
        for s in segments:
            segment_refs.append({
                'page': s.page,
                'seg_type': s.seg_type,
                'bbox': s.bbox,
                'text_preview': s.text[:50] + "..." if len(s.text) > 50 else s.text
            })
            
        return Chunk(
            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
            doc_id=doc_id,
            text=text,
            segments=segment_refs,
            page_start=page_start,
            page_end=page_end,
            chunk_type=chunk_type,
            metadata={}
        )

if __name__ == "__main__":
    import os
    import sys
    # Adding parent dir to path for smoke test
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    
    # Fake Segment class if not available for smoke test
    try:
        from src.ingestion.segment_builder import Segment
    except ImportError:
        @dataclass
        class Segment:
            page: int
            y0: float
            seg_type: str
            text: str
            bbox: Optional[tuple] = None
            table_id: Optional[int] = None
            
        sys.modules['src.ingestion.segment_builder'] = sys.modules[__name__]
    
    chunker = HybridChunker()
    segs = [
        Segment(1, 10.0, 'heading', "Introduction to AI"),
        Segment(1, 20.0, 'prose', "Artificial intelligence is a fascinating field. It involves machine learning and deep learning."),
        Segment(1, 30.0, 'prose', "Neural networks are models inspired by the human brain. They require a lot of data."),
        Segment(1, 40.0, 'table_row', "Model | Accuracy", table_id=1),
        Segment(1, 50.0, 'table_row', "CNN | 95%", table_id=1),
        Segment(2, 10.0, 'image_caption', "Figure 1: AI Architecture diagram")
    ]
    
    chunks = chunker.chunk_segments(segs, doc_id="doc_123")
    for c in chunks:
        print(f"Chunk ID: {c.chunk_id}")
        print(f"Type: {c.chunk_type}, Pages: {c.page_start}-{c.page_end}")
        print(f"Text:\n{c.text}\n")
