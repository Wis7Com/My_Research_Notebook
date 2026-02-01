"""Simplified PDF parser using PyMuPDF4LLM with page-level chunking."""

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf4llm


@dataclass
class PageChunk:
    """A single page chunk from a PDF document."""

    content: str
    page_number: int
    doc_title: str
    source_path: str
    parent_folder: str = ""
    metadata: dict = field(default_factory=dict)


class SimplePDFParser:
    """
    Simplified PDF parser using PyMuPDF4LLM.

    Uses page_chunks=True for accurate page-level tracking without
    heavy ML models or complex processing.
    """

    def parse(self, pdf_path: Path) -> list[PageChunk]:
        """
        Parse a PDF file into page-level chunks.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of PageChunk objects, one per page
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            pages = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=True,
                write_images=False,
            )
        except Exception as e:
            raise ValueError(f"Failed to parse PDF {pdf_path}: {e}") from e

        title = self._extract_title(pdf_path, pages)
        parent_folder = pdf_path.parent.name

        chunks = []
        for page_data in pages:
            content = page_data.get("text", "").strip()
            if not content:
                continue

            page_num = page_data.get("metadata", {}).get("page", 1)

            chunk = PageChunk(
                content=content,
                page_number=page_num,
                doc_title=title,
                source_path=str(pdf_path),
                parent_folder=parent_folder,
                metadata={
                    "parser": "simple",
                    "total_pages": len(pages),
                },
            )
            chunks.append(chunk)

        return chunks

    def _extract_title(self, pdf_path: Path, pages: list[dict]) -> str:
        """Extract document title from filename or first page."""
        filename = pdf_path.stem

        cleaned = filename.replace("_", " ").replace("-", " ")

        if pages:
            first_page = pages[0].get("text", "")
            if first_page:
                lines = first_page.strip().split("\n")
                for line in lines[:5]:
                    line = line.strip()
                    if line and len(line) > 10 and not line.startswith("#"):
                        if line.startswith("**") and line.endswith("**"):
                            return line.strip("*").strip()
                        if line[0].isupper():
                            return line[:100]

        return cleaned
