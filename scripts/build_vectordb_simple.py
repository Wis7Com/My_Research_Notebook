"""Build the vector database using simplified PyMuPDF4LLM parser."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from app.config import get_settings
from app.parsers import FastPDFParser, SimpleChunker
from app.vectordb import ChromaClient
from app.citations import BibTeXManager


def build_vectordb_simple(clear_existing: bool = False) -> None:
    """
    Build the vector database from all source PDFs using simplified parser.

    Args:
        clear_existing: Whether to clear existing data
    """
    settings = get_settings()
    settings.ensure_directories()

    print("=" * 60)
    print("Knowledge Base Builder")
    print("=" * 60)
    print("Using pure PyMuPDF text extraction")
    print("  - Fast and reliable")
    print("  - Character-based chunking")
    print("=" * 60)

    start_time = time.time()

    chroma_client = ChromaClient()
    bibtex = BibTeXManager()
    parser = FastPDFParser()
    chunker = SimpleChunker(
        chunk_size=2000,
        chunk_overlap=200,
        min_chunk_size=100,
    )

    if clear_existing:
        print("\nClearing existing data...")
        chroma_client.clear()

    pdf_files = list(settings.sources_dir.rglob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF files")

    pptx_files = list(settings.sources_dir.rglob("*.pptx"))
    if pptx_files:
        print(f"Found {len(pptx_files)} PPTX files (skipped - PDF only)")

    successful = 0
    failed = 0
    total_chunks = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            page_chunks = parser.parse(pdf_path)

            if not page_chunks:
                print(f"\n  Warning: No content from {pdf_path.name}")
                failed += 1
                continue

            chunks = chunker.chunk_pages(page_chunks)

            if not chunks:
                print(f"\n  Warning: No chunks from {pdf_path.name}")
                failed += 1
                continue

            try:
                chroma_client.add_simple_chunks(chunks, show_progress=False)
            except Exception as add_error:
                if "duplicate" in str(add_error).lower():
                    print(f"\n  Warning: Skipping duplicate chunks in {pdf_path.name}")
                    failed += 1
                    continue
                raise

            doc_id = chunks[0].doc_id
            title = chunks[0].title
            total_pages = page_chunks[0].metadata.get("total_pages", len(page_chunks))

            bibtex.add_document(
                doc_id=doc_id,
                title=title,
                source_path=str(pdf_path),
                parent_folder=pdf_path.parent.name,
                page_count=total_pages,
            )

            successful += 1
            total_chunks += len(chunks)

        except Exception as e:
            print(f"\n  Error processing {pdf_path.name}: {e}")
            failed += 1

    bibtex.save()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Successful: {successful} documents")
    print(f"Failed: {failed} documents")
    print(f"Total chunks: {total_chunks}")

    stats = chroma_client.get_stats()
    print(f"\nDatabase stats:")
    print(f"  - Documents: {stats['total_documents']}")
    print(f"  - Chunks: {stats['total_chunks']}")
    print(f"  - Location: {stats['persist_directory']}")

    citations = bibtex.list_entries()
    print(f"  - Citations: {len(citations)}")

    if successful > 0:
        avg_time = elapsed / successful
        print(f"\nAverage time per document: {avg_time:.2f} seconds")


def main():
    """Entry point with CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build vector database using simplified PyMuPDF4LLM parser"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing database before building",
    )

    args = parser.parse_args()

    build_vectordb_simple(clear_existing=args.clear)


if __name__ == "__main__":
    main()
