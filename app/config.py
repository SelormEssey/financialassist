"""Application settings loaded from environment variables and a local .env file."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated service configuration with safe defaults."""

    model_config = SettingsConfigDict(
        env_prefix="FINANCIALASSIST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "financialassist"
    debug: bool = False
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
