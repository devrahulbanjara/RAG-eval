from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = Field(
        "rag-eval-project", pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", max_length=45
    )
    PINECONE_API_KEY: str
    GROQ_API_KEY: str
    GEMINI_API_KEY: str

    DISPLAY_RETRIEVED_CHUNKS: bool = True

    model_config = SettingsConfigDict(env_file=".env")


setting = Settings()
