"""Hybrid PDF parser: pymupdf4llm for small files, pure PyMuPDF for large files."""

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
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


class HybridPDFParser:
    """
    Hybrid PDF parser using file size heuristic.

    - pymupdf4llm: Better layout analysis for smaller files (<3MB)
    - Pure PyMuPDF: Fast extraction for larger files (presentations, etc.)

    Large files typically have many images which slow down pymupdf4llm
    significantly without adding much value for text extraction.
    """

    # Files larger than this use fast fallback (1MB default)
    SIZE_THRESHOLD_MB = 1.0

    def __init__(self, size_threshold_mb: float = 1.0):
        """
        Initialize parser.

        Args:
            size_threshold_mb: Files larger than this (in MB) use fast parser
        """
        self.size_threshold_bytes = int(size_threshold_mb * 1024 * 1024)

    def parse(self, pdf_path: Path) -> list[PageChunk]:
        """
        Parse a PDF file, choosing parser based on file size.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of PageChunk objects, one per page
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        file_size = pdf_path.stat().st_size

        # Large files: use fast parser directly
        if file_size > self.size_threshold_bytes:
            return self._parse_with_fitz(pdf_path)

        # Small files: try pymupdf4llm first
        try:
            return self._parse_with_pymupdf4llm(pdf_path)
        except Exception:
            # Fallback to pure PyMuPDF on any error
            return self._parse_with_fitz(pdf_path)

    def _parse_with_pymupdf4llm(self, pdf_path: Path) -> list[PageChunk]:
        """Parse using pymupdf4llm (layout-aware)."""
        pages = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            write_images=False,
        )

        title = self._extract_title_from_pages(pdf_path, pages)
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
                    "parser": "pymupdf4llm",
                    "total_pages": len(pages),
                },
            )
            chunks.append(chunk)

        return chunks

    def _parse_with_fitz(self, pdf_path: Path) -> list[PageChunk]:
        """Parse using pure PyMuPDF (fast fallback)."""
        chunks = []
        doc = None

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            title = self._extract_title_from_doc(pdf_path, doc)
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
                        "parser": "fitz_fallback",
                        "total_pages": total_pages,
                    },
                )
                chunks.append(chunk)

        finally:
            if doc:
                doc.close()

        return chunks

    def _extract_title_from_pages(self, pdf_path: Path, pages: list[dict]) -> str:
        """Extract title from pymupdf4llm output."""
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

    def _extract_title_from_doc(self, pdf_path: Path, doc: fitz.Document) -> str:
        """Extract title from fitz document."""
        metadata = doc.metadata
        if metadata and metadata.get("title"):
            title = metadata["title"].strip()
            if len(title) > 5:
                return title

        filename = pdf_path.stem
        cleaned = filename.replace("_", " ").replace("-", " ")

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
