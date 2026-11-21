from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# LangSmith and the OpenAI SDK read their settings straight from the environment.
load_dotenv()


class Settings(BaseSettings):
    APP_NAME: str = Field(
        "rag-eval-project", pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", max_length=45
    )
    PINECONE_API_KEY: str
    OPENAI_API_KEY: str

    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_AGENT_ID: str = "insurance-rag"
    LANGSMITH_AGENT_ENVIRONMENT: str = "local"

    DISPLAY_RETRIEVED_CHUNKS: bool = True

    model_config = SettingsConfigDict(env_file=".env")


setting = Settings()
