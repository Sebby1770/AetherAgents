from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    LITELLM_MODEL: str = "gpt-4o"
    CHROMA_PATH: str = "./chroma_db"
    OPENAI_API_KEY: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()