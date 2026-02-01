"""Add individual documents to the vector database."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.parsers import SimplePDFParser, SimpleChunker
from app.vectordb import ChromaClient
from app.citations import BibTeXManager


def add_document(file_path: str) -> bool:
    """
    Add a single document to the vector database.

    Args:
        file_path: Path to the PDF file

    Returns:
        True if successful
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return False

    if file_path.suffix.lower() != ".pdf":
        print(f"Error: Only PDF files supported: {file_path}")
        return False

    settings = get_settings()
    settings.ensure_directories()

    chroma_client = ChromaClient()
    bibtex = BibTeXManager()
    parser = SimplePDFParser()
    chunker = SimpleChunker(
        chunk_size=2000,
        chunk_overlap=200,
        min_chunk_size=100,
    )

    print(f"Processing: {file_path.name}")

    try:
        page_chunks = parser.parse(file_path)

        if not page_chunks:
            print("  Error: Failed to parse PDF")
            return False

        title = page_chunks[0].doc_title
        total_pages = page_chunks[0].metadata.get("total_pages", len(page_chunks))

        print(f"  Title: {title}")
        print(f"  Pages: {total_pages}")

        chunks = chunker.chunk_pages(page_chunks)
        print(f"  Chunks: {len(chunks)}")

        chroma_client.add_simple_chunks(chunks)

        doc_id = chunks[0].doc_id
        key = bibtex.add_document(
            doc_id=doc_id,
            title=title,
            source_path=str(file_path),
            parent_folder=file_path.parent.name,
            page_count=total_pages,
        )
        bibtex.save()

        print(f"  Citation key: {key}")
        print("  Success!")

        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Entry point with CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Add a document to the vector database"
    )
    parser.add_argument(
        "file",
        help="Path to PDF file",
    )

    args = parser.parse_args()

    success = add_document(file_path=args.file)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
