"""MCP Server for Claude Code CLI integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from app.vectordb import ChromaClient, DocumentRetriever
from app.citations import BibTeXManager
from app.config import get_settings


server = Server("research-rag")


def get_retriever() -> DocumentRetriever:
    """Get or create retriever instance."""
    return DocumentRetriever()


def get_bibtex_manager() -> BibTeXManager:
    """Get or create BibTeX manager instance."""
    return BibTeXManager()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_documents",
            description="Search the research vector database for relevant content. Returns chunks with citations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for semantic search",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results (default: 5, max: 20)",
                        "default": 5,
                    },
                    "filter_phase": {
                        "type": "string",
                        "description": "Filter by folder name (partial match)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_citation",
            description="Get BibTeX citation for a document by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID from search results",
                    },
                },
                "required": ["doc_id"],
            },
        ),
        Tool(
            name="verify_citation",
            description="Verify a statement against source documents. Checks if the statement is supported by the sources.",
            inputSchema={
                "type": "object",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "Statement to verify",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Optional: specific document ID to check against",
                    },
                },
                "required": ["statement"],
            },
        ),
        Tool(
            name="list_documents",
            description="List all documents in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_document_context",
            description="Get full context from a specific document including all its sections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID",
                    },
                },
                "required": ["doc_id"],
            },
        ),
        Tool(
            name="get_database_stats",
            description="Get statistics about the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "search_documents":
        return await handle_search(arguments)
    elif name == "get_citation":
        return await handle_get_citation(arguments)
    elif name == "verify_citation":
        return await handle_verify_citation(arguments)
    elif name == "list_documents":
        return await handle_list_documents()
    elif name == "get_document_context":
        return await handle_get_document_context(arguments)
    elif name == "get_database_stats":
        return await handle_get_stats()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_search(arguments: dict) -> list[TextContent]:
    """Handle search_documents tool."""
    query = arguments.get("query", "")
    n_results = min(arguments.get("n_results", 5), 20)
    filter_phase = arguments.get("filter_phase")

    retriever = get_retriever()
    results = retriever.search(
        query=query,
        n_results=n_results,
        filter_phase=filter_phase,
    )

    if not results:
        return [TextContent(
            type="text",
            text="No results found for the query."
        )]

    output_parts = [f"Found {len(results)} results for: '{query}'\n"]

    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        relevance = result.get("relevance", 0)

        output_parts.append(f"\n--- Result {i} (relevance: {relevance:.2f}) ---")
        output_parts.append(f"Document: {metadata.get('title', 'Unknown')}")
        output_parts.append(f"Doc ID: {metadata.get('doc_id', 'N/A')}")

        page = metadata.get("page_start")
        if page and page > 0:
            output_parts.append(f"Page: {page}")

        section = metadata.get("section")
        if section:
            output_parts.append(f"Section: {section}")

        output_parts.append(f"\nContent:\n{result['content']}")

        footnotes = metadata.get("footnotes", {})
        if footnotes:
            output_parts.append("\nFootnotes:")
            for fn_num, fn_text in footnotes.items():
                output_parts.append(f"  [{fn_num}] {fn_text}")

    return [TextContent(type="text", text="\n".join(output_parts))]


async def handle_get_citation(arguments: dict) -> list[TextContent]:
    """Handle get_citation tool."""
    doc_id = arguments.get("doc_id", "")

    bibtex = get_bibtex_manager()
    key = bibtex.find_by_doc_id(doc_id)

    if not key:
        retriever = get_retriever()
        summary = retriever.get_document_summary(doc_id)

        if "error" in summary:
            return [TextContent(
                type="text",
                text=f"Document not found: {doc_id}"
            )]

        key = bibtex.add_document(
            doc_id=doc_id,
            title=summary["title"],
            source_path=summary["source_path"],
        )
        bibtex.save()

    entry = bibtex.get_bibtex_entry(key)
    citation = bibtex.get_citation(key)

    return [TextContent(
        type="text",
        text=f"Citation: {citation}\n\nBibTeX:\n{entry}"
    )]


async def handle_verify_citation(arguments: dict) -> list[TextContent]:
    """Handle verify_citation tool."""
    statement = arguments.get("statement", "")
    doc_id = arguments.get("doc_id")

    retriever = get_retriever()

    results = retriever.search(
        query=statement,
        n_results=5,
        filter_doc_id=doc_id,
    )

    if not results:
        return [TextContent(
            type="text",
            text="No supporting sources found for this statement."
        )]

    output_parts = ["Verification Results:\n"]

    for i, result in enumerate(results, 1):
        relevance = result.get("relevance", 0)
        metadata = result["metadata"]

        support_level = "STRONG" if relevance > 0.8 else "MODERATE" if relevance > 0.6 else "WEAK"

        output_parts.append(f"\n--- Source {i} ({support_level} support, {relevance:.2f}) ---")
        output_parts.append(f"Document: {metadata.get('title', 'Unknown')}")
        output_parts.append(f"Doc ID: {metadata.get('doc_id', 'N/A')}")

        page = metadata.get("page_start")
        if page and page > 0:
            output_parts.append(f"Page: {page}")

        output_parts.append(f"\nRelevant content:\n{result['content'][:1000]}...")

    return [TextContent(type="text", text="\n".join(output_parts))]


async def handle_list_documents() -> list[TextContent]:
    """Handle list_documents tool."""
    client = ChromaClient()
    documents = client.list_documents()

    if not documents:
        return [TextContent(
            type="text",
            text="No documents in the knowledge base. Run build_vectordb.py first."
        )]

    output_parts = [f"Documents in knowledge base ({len(documents)} total):\n"]

    docs_by_folder = {}
    for doc in documents:
        folder = doc.get("parent_folder", "Other")
        if folder not in docs_by_folder:
            docs_by_folder[folder] = []
        docs_by_folder[folder].append(doc)

    for folder in sorted(docs_by_folder.keys()):
        output_parts.append(f"\n## {folder}")
        for doc in docs_by_folder[folder]:
            output_parts.append(
                f"  - {doc['title']} (ID: {doc['doc_id']}, {doc['total_chunks']} chunks)"
            )

    return [TextContent(type="text", text="\n".join(output_parts))]


async def handle_get_document_context(arguments: dict) -> list[TextContent]:
    """Handle get_document_context tool."""
    doc_id = arguments.get("doc_id", "")

    client = ChromaClient()
    chunks = client.get_document_chunks(doc_id)

    if not chunks:
        return [TextContent(
            type="text",
            text=f"Document not found: {doc_id}"
        )]

    first_metadata = chunks[0]["metadata"]
    output_parts = [
        f"Document: {first_metadata.get('title', 'Unknown')}",
        f"Total chunks: {len(chunks)}",
        f"Source: {first_metadata.get('source_path', 'N/A')}",
        "\n--- Full Content ---\n",
    ]

    for chunk in chunks:
        section = chunk["metadata"].get("section")
        if section:
            output_parts.append(f"\n## {section}\n")
        output_parts.append(chunk["content"])

    return [TextContent(type="text", text="\n".join(output_parts))]


async def handle_get_stats() -> list[TextContent]:
    """Handle get_database_stats tool."""
    client = ChromaClient()
    stats = client.get_stats()

    output = f"""Knowledge Base Statistics:
- Total documents: {stats['total_documents']}
- Total chunks: {stats['total_chunks']}
- Collection: {stats['collection_name']}
- Storage: {stats['persist_directory']}"""

    return [TextContent(type="text", text=output)]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
