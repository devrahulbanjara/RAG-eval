from groq import Groq
from sentence_transformers import SentenceTransformer

from .config import setting
from .index_to_pinecone import (
    EMBEDDING_MODEL,
    NAMESPACE,
    QUERY_PREFIX,
    get_pinecone_index,
)
from .prompts import ANSWER_PROMPT, SYSTEM_PROMPT

LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
SEPARATOR = "=" * 80

_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    # Loading the model takes a few seconds and the evals call retrieve() once
    # per golden, so keep one instance for the life of the process.
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k chunks for a query, best match first."""
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


def to_context(chunks: list[dict]) -> list[str]:
    """Label each chunk with the policy it came from.

    The corpus holds three competing health policies and three near-identical
    NFIP forms, so an unlabelled context invites an answer that blends them.
    """
    return [f"[{chunk['source']}]\n{chunk['text'].strip()}" for chunk in chunks]


def generate(query: str, context: list[str]) -> str:
    """Answer a query from the given context blocks and nothing else."""
    client = Groq(api_key=setting.GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    context="\n\n".join(context), question=query
                ),
            },
        ],
    )
    return completion.choices[0].message.content


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
    print(generate(search_query, to_context(chunks)))


if __name__ == "__main__":
    main()
