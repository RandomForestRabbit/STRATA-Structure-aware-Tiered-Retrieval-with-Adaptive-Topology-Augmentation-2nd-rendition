import os
import re
from typing import List, Dict, Tuple
import anthropic
from src.retrieval.retriever import RetrievalResult

class Generator:
    def __init__(self, model='claude-sonnet-4-6', max_answer_tokens=1024, api_key=None):
        self.model = model
        self.max_answer_tokens = max_answer_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
    def generate(self, retrieval_result: RetrievalResult) -> dict:
        if not retrieval_result.chunks:
            return {
                'answer': 'I could not find sufficient information in the provided documents to answer this question.',
                'citations': [],
                'contested_facts': [],
                'model': self.model,
                'fallback_used': True
            }
            
        system_message, user_message = self._build_prompt(retrieval_result)
        
        if not self.api_key:
            return {
                'answer': 'API key not provided. Unable to call LLM.',
                'citations': [],
                'contested_facts': [],
                'model': self.model,
                'fallback_used': True
            }
            
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            # Claude models currently don't use a separate system message param in messages API 
            # for older models but for claude-3 they do. Assuming standard API usage:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_answer_tokens,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            answer = response.content[0].text
        except Exception as e:
            return {
                'answer': f'Error generating response: {str(e)}',
                'citations': [],
                'contested_facts': [],
                'model': self.model,
                'fallback_used': True
            }
            
        citations = self._parse_citations(answer)
        
        # Verify citations against provided chunks
        valid_citations = []
        provided_sources = {i+1: c for i, c in enumerate(retrieval_result.chunks)}
        for cit in citations:
            if cit['source_num'] in provided_sources:
                valid_citations.append(cit)
                
        contested_facts = self._check_contested_facts(answer, retrieval_result.chunks)
        
        return {
            'answer': answer,
            'citations': valid_citations,
            'contested_facts': contested_facts,
            'model': self.model,
            'fallback_used': False
        }
        
    def _build_prompt(self, retrieval_result: RetrievalResult) -> Tuple[str, str]:
        system_message = (
            "You are a precise research assistant. Answer ONLY using the provided context. "
            "Cite sources using [Source N] notation. If the context contains contested/contradictory "
            "facts, acknowledge both versions. If context is insufficient, say so."
        )
        
        context_parts = []
        for i, chunk in enumerate(retrieval_result.chunks):
            source_num = i + 1
            tag = chunk.get('source_tag', 'direct_vector')
            if tag == 'both':
                label = '[DIRECT] [GRAPH-HOP]'
            elif tag.startswith('graph'):
                label = '[GRAPH-HOP]'
            else:
                label = '[DIRECT]'
                
            doc_id = chunk.get('metadata', {}).get('doc_id', 'unknown')
            page = chunk.get('metadata', {}).get('page', 'unknown')
            
            context_parts.append(f"{label} [Source {source_num}: {doc_id}, p.{page}]\n{chunk['text']}")
            
        context_str = "\n\n".join(context_parts)
        user_message = f"Context:\n{context_str}\n\nQuery: {retrieval_result.query}"
        
        return system_message, user_message
        
    def _parse_citations(self, answer: str) -> List[dict]:
        pattern = r'\[Source (\d+)\]'
        matches = re.finditer(pattern, answer)
        citations = []
        seen = set()
        for match in matches:
            num = int(match.group(1))
            if num not in seen:
                citations.append({'source_num': num})
                seen.add(num)
        return citations
        
    def _check_contested_facts(self, answer: str, chunks: List[dict]) -> List[str]:
        # Simple placeholder for checking contested facts if any chunk was tagged as contested
        unaddressed = []
        for chunk in chunks:
            if chunk.get('source_tag') == 'contested':
                if chunk['text'].lower() not in answer.lower():
                    unaddressed.append("A contested fact was not addressed in the answer.")
        return unaddressed

if __name__ == "__main__":
    generator = Generator(api_key="dummy")
    result = RetrievalResult(
        chunks=[
            {'text': 'The sky is blue.', 'source_tag': 'direct_vector', 'metadata': {'doc_id': 'd1', 'page': 1}},
            {'text': 'The sky is green.', 'source_tag': 'contested', 'metadata': {'doc_id': 'd2', 'page': 2}}
        ],
        query="What color is the sky?",
        vector_results_count=1,
        graph_results_count=1,
        fused_count=2,
        reranked=False
    )
    sys, usr = generator._build_prompt(result)
    print("System Prompt:\n", sys)
    print("\nUser Prompt:\n", usr)
