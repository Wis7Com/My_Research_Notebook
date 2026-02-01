"""Fast PDF parser using pure PyMuPDF (fitz) for text extraction."""

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageChunk:
    """A single page chunk from a PDF document."""

    content: str
    page_number: int
    doc_title: str
    source_path: str
    parent_folder: str = ""
    metadata: dict = field(default_factory=dict)


class FastPDFParser:
    """
    Fast PDF parser using pure PyMuPDF text extraction.

    Much faster than pymupdf4llm as it skips layout analysis.
    Best for documents where text order is sufficient.
    """

    def __init__(self, timeout_seconds: int = 30):
        """Initialize parser with optional timeout per file."""
        self.timeout_seconds = timeout_seconds

    def parse(self, pdf_path: Path) -> list[PageChunk]:
        """
        Parse a PDF file into page-level chunks using fast text extraction.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of PageChunk objects, one per page
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        chunks = []
        doc = None

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            title = self._extract_title(pdf_path, doc)
            parent_folder = pdf_path.parent.name

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text")

                if not text or not text.strip():
                    continue

                chunk = PageChunk(
                    content=text.strip(),
                    page_number=page_num + 1,
                    doc_title=title,
                    source_path=str(pdf_path),
                    parent_folder=parent_folder,
                    metadata={
                        "parser": "fast",
                        "total_pages": total_pages,
                    },
                )
                chunks.append(chunk)

        except Exception as e:
            raise ValueError(f"Failed to parse PDF {pdf_path}: {e}") from e
        finally:
            if doc:
                doc.close()

        return chunks

    def _extract_title(self, pdf_path: Path, doc: fitz.Document) -> str:
        """Extract document title from metadata, filename, or first page."""
        # Try PDF metadata first
        metadata = doc.metadata
        if metadata and metadata.get("title"):
            title = metadata["title"].strip()
            if len(title) > 5:
                return title

        # Fall back to filename
        filename = pdf_path.stem
        cleaned = filename.replace("_", " ").replace("-", " ")

        # Try first page content
        if len(doc) > 0:
            first_page = doc[0]
            text = first_page.get_text("text")
            if text:
                lines = text.strip().split("\n")
                for line in lines[:5]:
                    line = line.strip()
                    if line and len(line) > 10:
                        if line[0].isupper():
                            return line[:100]

        return cleaned
