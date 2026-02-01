"""Simplified character-based chunker without tiktoken dependency."""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from .simple_parser import PageChunk


@dataclass
class SimpleChunk:
    """A text chunk with metadata for vector storage."""

    chunk_id: str
    content: str
    doc_id: str
    source_path: str
    title: str
    page_number: int
    chunk_index: int = 0
    total_chunks: int = 0
    char_count: int = 0
    parent_folder: str = ""
    metadata: dict = field(default_factory=dict)


class SimpleChunker:
    """
    Character-based document chunker.

    Splits text by paragraphs and combines them into chunks
    without token counting overhead.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to keep
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_pages(self, page_chunks: list[PageChunk]) -> list[SimpleChunk]:
        """
        Chunk a list of page chunks into smaller pieces.

        Args:
            page_chunks: List of PageChunk objects from SimplePDFParser

        Returns:
            List of SimpleChunk objects
        """
        if not page_chunks:
            return []

        first_page = page_chunks[0]
        doc_id = self._generate_doc_id(first_page.source_path)

        all_chunks = []

        for page_chunk in page_chunks:
            page_chunks_result = self._chunk_page(
                page_chunk=page_chunk,
                doc_id=doc_id,
                base_index=len(all_chunks),
            )
            all_chunks.extend(page_chunks_result)

        total_chunks = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total_chunks

        return all_chunks

    def _generate_doc_id(self, source_path: str) -> str:
        """Generate a unique document ID based on full path."""
        return hashlib.md5(source_path.encode()).hexdigest()[:12]

    def _chunk_page(
        self,
        page_chunk: PageChunk,
        doc_id: str,
        base_index: int,
    ) -> list[SimpleChunk]:
        """Chunk a single page's content."""
        content = page_chunk.content

        if len(content) <= self.chunk_size:
            return [
                self._create_chunk(
                    content=content,
                    doc_id=doc_id,
                    page_chunk=page_chunk,
                    index=base_index,
                )
            ]

        paragraphs = self._split_into_paragraphs(content)
        chunks = []
        current_text = ""

        for para in paragraphs:
            if len(para) > self.chunk_size:
                if current_text.strip():
                    chunks.append(
                        self._create_chunk(
                            content=current_text.strip(),
                            doc_id=doc_id,
                            page_chunk=page_chunk,
                            index=base_index + len(chunks),
                        )
                    )
                    current_text = ""

                sub_chunks = self._split_large_text(para)
                for sub_text in sub_chunks:
                    chunks.append(
                        self._create_chunk(
                            content=sub_text,
                            doc_id=doc_id,
                            page_chunk=page_chunk,
                            index=base_index + len(chunks),
                        )
                    )
                continue

            test_text = current_text + "\n\n" + para if current_text else para

            if len(test_text) > self.chunk_size:
                if current_text.strip():
                    chunks.append(
                        self._create_chunk(
                            content=current_text.strip(),
                            doc_id=doc_id,
                            page_chunk=page_chunk,
                            index=base_index + len(chunks),
                        )
                    )

                overlap_text = self._get_overlap_text(current_text)
                current_text = overlap_text + "\n\n" + para if overlap_text else para
            else:
                current_text = test_text

        if current_text.strip() and len(current_text.strip()) >= self.min_chunk_size:
            chunks.append(
                self._create_chunk(
                    content=current_text.strip(),
                    doc_id=doc_id,
                    page_chunk=page_chunk,
                    index=base_index + len(chunks),
                )
            )

        return chunks

    def _split_into_paragraphs(self, content: str) -> list[str]:
        """Split content into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", content)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_large_text(self, text: str) -> list[str]:
        """Split large text by sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sentence in sentences:
            test = current + " " + sentence if current else sentence
            if len(test) > self.chunk_size and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = test

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end."""
        if len(text) <= self.chunk_overlap:
            return text

        overlap_start = len(text) - self.chunk_overlap

        space_pos = text.find(" ", overlap_start)
        if space_pos != -1 and space_pos < len(text) - 50:
            return text[space_pos + 1 :]

        return text[-self.chunk_overlap :]

    def _create_chunk(
        self,
        content: str,
        doc_id: str,
        page_chunk: PageChunk,
        index: int,
    ) -> SimpleChunk:
        """Create a SimpleChunk object."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:6]
        chunk_id = f"{doc_id}_{index:04d}_{content_hash}"

        return SimpleChunk(
            chunk_id=chunk_id,
            content=content,
            doc_id=doc_id,
            source_path=page_chunk.source_path,
            title=page_chunk.doc_title,
            page_number=page_chunk.page_number,
            chunk_index=index,
            char_count=len(content),
            parent_folder=page_chunk.parent_folder,
            metadata={
                "parser": "simple",
                "total_pages": page_chunk.metadata.get("total_pages", 0),
            },
        )
