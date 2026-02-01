"""Main entry point for the Research RAG application."""

import sys
import argparse
from pathlib import Path


def cmd_build(args):
    """Build the vector database."""
    from scripts.build_vectordb_simple import build_vectordb_simple

    build_vectordb_simple(clear_existing=args.clear)


def cmd_add(args):
    """Add a document to the database."""
    from scripts.add_documents import add_document

    success = add_document(file_path=args.file)
    sys.exit(0 if success else 1)


def cmd_ui(args):
    """Launch the Gradio UI."""
    from app.ui import NotebookUI

    ui = NotebookUI()
    ui.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )


def cmd_mcp(args):
    """Run the MCP server."""
    import asyncio
    from mcp_server.server import main as mcp_main

    asyncio.run(mcp_main())


def cmd_search(args):
    """Search the database from command line."""
    from app.vectordb import DocumentRetriever

    retriever = DocumentRetriever()
    results = retriever.search(
        query=args.query,
        n_results=args.n,
        filter_phase=args.phase,
    )

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        relevance = result.get("relevance", 0)

        print(f"--- Result {i} (relevance: {relevance:.2f}) ---")
        print(f"Document: {metadata.get('title', 'Unknown')}")

        page = metadata.get("page_start")
        if page and page > 0:
            print(f"Page: {page}")

        print(f"\n{result['content'][:500]}...\n")


def cmd_stats(args):
    """Show database statistics."""
    from app.vectordb import ChromaClient
    from app.citations import BibTeXManager

    client = ChromaClient()
    bibtex = BibTeXManager()

    stats = client.get_stats()
    citations = bibtex.list_entries()

    print("Knowledge Base Statistics")
    print("=" * 40)
    print(f"Documents: {stats['total_documents']}")
    print(f"Chunks: {stats['total_chunks']}")
    print(f"Citations: {len(citations)}")
    print(f"Collection: {stats['collection_name']}")
    print(f"Location: {stats['persist_directory']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="My Research Notebook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build     Build vector database from source PDFs
  add       Add a single document to the database
  ui        Launch the Gradio web interface
  mcp       Run the MCP server for Claude Code
  search    Search the database from command line
  stats     Show database statistics

Examples:
  python -m app.main build --clear
  python -m app.main add path/to/document.pdf
  python -m app.main ui --port 8080
  python -m app.main search "search query"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    build_parser = subparsers.add_parser("build", help="Build vector database")
    build_parser.add_argument("--clear", action="store_true", help="Clear existing data")
    build_parser.set_defaults(func=cmd_build)

    add_parser = subparsers.add_parser("add", help="Add document")
    add_parser.add_argument("file", help="Path to PDF file")
    add_parser.set_defaults(func=cmd_add)

    ui_parser = subparsers.add_parser("ui", help="Launch web UI")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    ui_parser.add_argument("--port", type=int, default=7860, help="Port number")
    ui_parser.add_argument("--share", action="store_true", help="Create public link")
    ui_parser.set_defaults(func=cmd_ui)

    mcp_parser = subparsers.add_parser("mcp", help="Run MCP server")
    mcp_parser.set_defaults(func=cmd_mcp)

    search_parser = subparsers.add_parser("search", help="Search database")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", type=int, default=5, help="Number of results")
    search_parser.add_argument("--phase", help="Filter by phase")
    search_parser.set_defaults(func=cmd_search)

    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
