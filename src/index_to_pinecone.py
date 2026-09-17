from pinecone import Pinecone

from src.config import setting

pc: Pinecone = Pinecone(api_key=setting.PINECONE_API_KEY)

index_name = setting.APP_NAME

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
    )

index = pc.Index(name=index_name)
