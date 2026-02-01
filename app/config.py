"""Application configuration."""

from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # Chunking settings
    chunk_size: int = 1500
    chunk_overlap: int = 200
    min_chunk_size: int = 100

    # ChromaDB settings
    collection_name: str = "research_documents"

    # Retrieval settings
    default_n_results: int = 5
    max_n_results: int = 20

    # Paths (computed after init)
    project_root: Optional[Path] = None
    sources_dir: Optional[Path] = None
    data_dir: Optional[Path] = None
    parsed_markdown_dir: Optional[Path] = None
    chroma_db_dir: Optional[Path] = None
    citations_dir: Optional[Path] = None
    outputs_dir: Optional[Path] = None

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def set_derived_paths(self) -> "Settings":
        """Set derived paths after model initialization."""
        self.project_root = Path(__file__).parent.parent
        self.sources_dir = self.project_root / "sources_raw"
        self.data_dir = self.project_root / "data"
        self.parsed_markdown_dir = self.data_dir / "parsed_markdown"
        self.chroma_db_dir = self.data_dir / "chroma_db"
        self.citations_dir = self.data_dir / "citations"
        self.outputs_dir = self.project_root / "outputs"
        return self

    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        for dir_path in [
            self.parsed_markdown_dir,
            self.chroma_db_dir,
            self.citations_dir,
            self.outputs_dir / "drafts",
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
