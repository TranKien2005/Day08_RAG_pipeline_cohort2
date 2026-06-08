"""
Task 10 — Generation Có Citation.

Uses retrieval from Task 9, reorders context to reduce lost-in-the-middle, and
returns a Vietnamese answer with citations. If no generation API key is present,
it falls back to an extractive cited answer so tests and demos still run offline.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

# User-selected generation model via OpenAI-compatible router. The model can be
# overridden in .env with GENERATION_MODEL. If no key is present, generation uses
# a local extractive fallback with citations.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "groq/llama-3.3-70b-versatile")
GENERATION_BASE_URL = os.getenv("GENERATION_BASE_URL", "https://api.9router.com/v1")


SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Sắp xếp chunks để tránh lost-in-the-middle effect."""
    if len(chunks) <= 2:
        return chunks

    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    last_even_index = len(chunks) - 1 if (len(chunks) - 1) % 2 == 1 else len(chunks) - 2
    for i in range(last_even_index, 0, -2):
        reordered.append(chunks[i])

    return reordered


def _citation_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("path") or f"Source {index}"
    return source.replace(".md", "").replace(".html", "").replace(".pdf", "").replace(".docx", "")


def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string cho prompt, kèm source labels."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = _citation_label(chunk, i)
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


def _get_generation_api_key() -> str | None:
    return os.getenv("NINEROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def _call_llm(query: str, context: str) -> str | None:
    api_key = _get_generation_api_key()
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=GENERATION_BASE_URL)
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_parts = ["Dựa trên các nguồn đã truy xuất, thông tin liên quan gồm:"]
    for i, chunk in enumerate(chunks[:3], 1):
        source = _citation_label(chunk, i)
        snippet = " ".join(chunk.get("content", "").split())[:350]
        answer_parts.append(f"- {snippet} [{source}]")
    return "\n".join(answer_parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """End-to-end RAG generation có citation."""
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = _call_llm(query, context)
    if not answer:
        answer = _extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
