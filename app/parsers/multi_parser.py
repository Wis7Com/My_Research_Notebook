"""Multi-format document parser supporting PDF, MD, TXT, XML, DOCX."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


@dataclass
class PageChunk:
    """A single page/section chunk from a document."""

    content: str
    page_number: int
    doc_title: str
    source_path: str
    parent_folder: str = ""
    metadata: dict = field(default_factory=dict)


class MultiFormatParser:
    """
    Parser supporting multiple document formats.

    Supported formats:
    - PDF (.pdf)
    - Markdown (.md)
    - Plain text (.txt)
    - XML (.xml)
    - Word documents (.docx)
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".xml", ".docx"}

    def parse(self, file_path: Path) -> list[PageChunk]:
        """
        Parse a document file into chunks.

        Args:
            file_path: Path to the document file

        Returns:
            List of PageChunk objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {ext}")

        if ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext == ".md":
            return self._parse_markdown(file_path)
        elif ext == ".txt":
            return self._parse_text(file_path)
        elif ext == ".xml":
            return self._parse_xml(file_path)
        elif ext == ".docx":
            return self._parse_docx(file_path)

        return []

    def _parse_pdf(self, file_path: Path) -> list[PageChunk]:
        """Parse PDF using PyMuPDF."""
        chunks = []
        doc = None

        try:
            doc = fitz.open(str(file_path))
            total_pages = len(doc)
            title = self._get_title(file_path, doc.metadata.get("title"))
            parent_folder = file_path.parent.name

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text")

                if not text or not text.strip():
                    continue

                chunk = PageChunk(
                    content=text.strip(),
                    page_number=page_num + 1,
                    doc_title=title,
                    source_path=str(file_path),
                    parent_folder=parent_folder,
                    metadata={
                        "parser": "pdf",
                        "total_pages": total_pages,
                    },
                )
                chunks.append(chunk)

        finally:
            if doc:
                doc.close()

        return chunks

    def _parse_markdown(self, file_path: Path) -> list[PageChunk]:
        """Parse Markdown file."""
        content = file_path.read_text(encoding="utf-8")
        title = self._get_title(file_path)
        parent_folder = file_path.parent.name

        # Split by headers for better chunking
        sections = self._split_by_headers(content)

        chunks = []
        for i, section in enumerate(sections, 1):
            if not section.strip():
                continue

            chunk = PageChunk(
                content=section.strip(),
                page_number=i,
                doc_title=title,
                source_path=str(file_path),
                parent_folder=parent_folder,
                metadata={
                    "parser": "markdown",
                    "total_pages": len(sections),
                },
            )
            chunks.append(chunk)

        return chunks if chunks else [self._single_chunk(file_path, content, "markdown")]

    def _parse_text(self, file_path: Path) -> list[PageChunk]:
        """Parse plain text file."""
        content = file_path.read_text(encoding="utf-8")
        title = self._get_title(file_path)
        parent_folder = file_path.parent.name

        # Split by double newlines for paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        # Group paragraphs into reasonable chunks (roughly 2000 chars each)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_num = 1

        for para in paragraphs:
            if current_size + len(para) > 2000 and current_chunk:
                chunk = PageChunk(
                    content="\n\n".join(current_chunk),
                    page_number=chunk_num,
                    doc_title=title,
                    source_path=str(file_path),
                    parent_folder=parent_folder,
                    metadata={
                        "parser": "text",
                        "total_pages": 0,  # Will update later
                    },
                )
                chunks.append(chunk)
                current_chunk = []
                current_size = 0
                chunk_num += 1

            current_chunk.append(para)
            current_size += len(para)

        # Add remaining content
        if current_chunk:
            chunk = PageChunk(
                content="\n\n".join(current_chunk),
                page_number=chunk_num,
                doc_title=title,
                source_path=str(file_path),
                parent_folder=parent_folder,
                metadata={
                    "parser": "text",
                    "total_pages": chunk_num,
                },
            )
            chunks.append(chunk)

        # Update total_pages
        for chunk in chunks:
            chunk.metadata["total_pages"] = len(chunks)

        return chunks if chunks else [self._single_chunk(file_path, content, "text")]

    def _parse_xml(self, file_path: Path) -> list[PageChunk]:
        """Parse XML file - extract text content."""
        import re

        content = file_path.read_text(encoding="utf-8")
        title = self._get_title(file_path)
        parent_folder = file_path.parent.name

        # Remove XML tags but keep text content
        text_content = re.sub(r"<[^>]+>", " ", content)
        text_content = re.sub(r"\s+", " ", text_content).strip()

        if not text_content:
            return []

        chunk = PageChunk(
            content=text_content,
            page_number=1,
            doc_title=title,
            source_path=str(file_path),
            parent_folder=parent_folder,
            metadata={
                "parser": "xml",
                "total_pages": 1,
            },
        )
        return [chunk]

    def _parse_docx(self, file_path: Path) -> list[PageChunk]:
        """Parse Word document using python-docx."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx is required for .docx support. Install with: pip install python-docx")

        doc = Document(str(file_path))
        title = self._get_title(file_path)
        parent_folder = file_path.parent.name

        # Extract paragraphs
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            return []

        # Group into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_num = 1

        for para in paragraphs:
            if current_size + len(para) > 2000 and current_chunk:
                chunk = PageChunk(
                    content="\n\n".join(current_chunk),
                    page_number=chunk_num,
                    doc_title=title,
                    source_path=str(file_path),
                    parent_folder=parent_folder,
                    metadata={
                        "parser": "docx",
                        "total_pages": 0,
                    },
                )
                chunks.append(chunk)
                current_chunk = []
                current_size = 0
                chunk_num += 1

            current_chunk.append(para)
            current_size += len(para)

        if current_chunk:
            chunk = PageChunk(
                content="\n\n".join(current_chunk),
                page_number=chunk_num,
                doc_title=title,
                source_path=str(file_path),
                parent_folder=parent_folder,
                metadata={
                    "parser": "docx",
                    "total_pages": chunk_num,
                },
            )
            chunks.append(chunk)

        for chunk in chunks:
            chunk.metadata["total_pages"] = len(chunks)

        return chunks

    def _split_by_headers(self, content: str) -> list[str]:
        """Split markdown content by headers."""
        import re

        # Split by markdown headers (# ## ### etc.)
        sections = re.split(r"\n(?=#{1,6}\s)", content)
        return [s for s in sections if s.strip()]

    def _get_title(self, file_path: Path, metadata_title: Optional[str] = None) -> str:
        """Extract title from metadata or filename."""
        if metadata_title and len(metadata_title.strip()) > 3:
            return metadata_title.strip()

        return file_path.stem.replace("_", " ").replace("-", " ")

    def _single_chunk(self, file_path: Path, content: str, parser_type: str) -> PageChunk:
        """Create a single chunk for the entire content."""
        return PageChunk(
            content=content.strip(),
            page_number=1,
            doc_title=self._get_title(file_path),
            source_path=str(file_path),
            parent_folder=file_path.parent.name,
            metadata={
                "parser": parser_type,
                "total_pages": 1,
            },
        )
