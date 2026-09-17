from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import UnauthorizedException
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .config import setting

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
MAX_TOKENS = 512

# Queries must carry this prefix for bge models; documents must not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_TOKENS = 480
CHUNK_OVERLAP = 50

NAMESPACE = "bge-small-480tok"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def get_pinecone_index(index_name: str):
    pc = Pinecone(api_key=setting.PINECONE_API_KEY)
    try:
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
    except UnauthorizedException as exc:
        raise RuntimeError("PINECONE_API_KEY is missing or invalid") from exc
    return pc.Index(name=index_name)


def load_chunks(data_dir: Path) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[dict] = []
    for markdown_file in sorted(data_dir.glob("*.md")):
        sections = header_splitter.split_text(markdown_file.read_text(encoding="utf-8"))
        for piece in size_splitter.split_documents(sections):
            chunks.append(
                {
                    "_id": f"rec{len(chunks) + 1}",
                    "chunk_text": piece.page_content,
                    "source": markdown_file.stem,
                }
            )
    return chunks


def main() -> None:
    chunks = load_chunks(DATA_DIR)
    print(f"{len(chunks)} chunks from {DATA_DIR}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(
        [chunk["chunk_text"] for chunk in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    index = get_pinecone_index(setting.APP_NAME)
    index.upsert(
        vectors=[
            {
                "id": chunk["_id"],
                "values": vector.tolist(),
                "metadata": {
                    "chunk_text": chunk["chunk_text"],
                    "source": chunk["source"],
                },
            }
            for chunk, vector in zip(chunks, vectors)
        ],
        namespace=NAMESPACE,
        batch_size=100,
    )
    print(f"upserted {len(chunks)} vectors into namespace {NAMESPACE!r}")


if __name__ == "__main__":
    main()
