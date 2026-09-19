from sentence_transformers import SentenceTransformer

from .config import setting
from .index_to_pinecone import (
    EMBEDDING_MODEL,
    NAMESPACE,
    QUERY_PREFIX,
    get_pinecone_index,
)
from .llm import generate_response_from_context

TOP_K = 5
SEPARATOR = "=" * 80

_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    query_vector = _get_embedding_model().encode(
        QUERY_PREFIX + query, normalize_embeddings=True
    )
    index = get_pinecone_index(setting.APP_NAME)
    matches = index.query(
        namespace=NAMESPACE,
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
    )["matches"]
    return [
        {
            "text": match["metadata"]["chunk_text"],
            "source": match["metadata"]["source"],
            "score": match["score"],
        }
        for match in matches
    ]


def label_chunks_with_source(chunks: list[dict]) -> list[str]:
    return [f"[{chunk['source']}]\n{chunk['text'].strip()}" for chunk in chunks]


def display_chunks(query: str, chunks: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print(f"{len(chunks)} chunks retrieved from namespace {NAMESPACE!r}\n")

    for position, chunk in enumerate(chunks, start=1):
        print(SEPARATOR)
        print(f"Chunk {position}  |  score {chunk['score']:.4f}  |  {chunk['source']}")
        print(SEPARATOR)
        print()
        print(chunk["text"].strip())
        print()


def main() -> None:
    search_query = "Is knee replacement surgery covered by the policy?"

    chunks = retrieve(search_query)
    if setting.DISPLAY_RETRIEVED_CHUNKS:
        display_chunks(search_query, chunks)

    print(SEPARATOR)
    print("Answer")
    print(SEPARATOR)
    print()
    print(
        generate_response_from_context(search_query, label_chunks_with_source(chunks))
    )


if __name__ == "__main__":
    main()
