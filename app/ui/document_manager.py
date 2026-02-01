"""Gradio document manager UI."""

from pathlib import Path
from typing import Optional
import shutil

import gradio as gr

from ..config import get_settings
from ..parsers import PyMuPDFParser, DocumentChunker, get_docling_parser
from ..vectordb import ChromaClient, DocumentRetriever
from ..citations import BibTeXManager


class DocumentManagerUI:
    """Gradio-based document management interface."""

    def __init__(self):
        """Initialize the UI components."""
        self.settings = get_settings()
        self.settings.ensure_directories()

        self.chroma_client = ChromaClient()
        self.retriever = DocumentRetriever(self.chroma_client)
        self.bibtex = BibTeXManager()
        self.chunker = DocumentChunker()

        self.docling_parser = None
        self.pymupdf_parser = None

    def _get_parser(self, use_fallback: bool = False):
        """Get PDF parser instance."""
        if use_fallback:
            if not self.pymupdf_parser:
                self.pymupdf_parser = PyMuPDFParser()
            return self.pymupdf_parser

        if not self.docling_parser:
            try:
                DoclingParser = get_docling_parser()
                self.docling_parser = DoclingParser()
            except Exception:
                if not self.pymupdf_parser:
                    self.pymupdf_parser = PyMuPDFParser()
                return self.pymupdf_parser
        return self.docling_parser

    def upload_document(
        self,
        file_path: str,
        progress: Optional[gr.Progress] = None,
    ) -> str:
        """
        Upload and process a PDF document.

        Args:
            file_path: Path to uploaded file
            progress: Gradio progress tracker

        Returns:
            Status message
        """
        if not file_path:
            return "No file selected."

        file_path = Path(file_path)
        if not file_path.suffix.lower() == ".pdf":
            return "Only PDF files are supported."

        try:
            dest_path = self.settings.sources_dir / file_path.name
            shutil.copy(file_path, dest_path)

            if progress:
                progress(0.2, desc="Parsing PDF...")

            parser = self._get_parser()
            doc = parser.parse(dest_path)

            if not doc:
                parser = self._get_parser(use_fallback=True)
                doc = parser.parse(dest_path)

            if not doc:
                return f"Failed to parse PDF: {file_path.name}"

            if progress:
                progress(0.5, desc="Chunking document...")

            chunks = self.chunker.chunk_document(doc)

            if progress:
                progress(0.7, desc="Adding to vector database...")

            self.chroma_client.add_chunks(chunks, show_progress=False)

            if progress:
                progress(0.9, desc="Creating citation...")

            doc_id = chunks[0].doc_id if chunks else "unknown"
            self.bibtex.add_document(
                doc_id=doc_id,
                title=doc.title,
                source_path=str(dest_path),
                parent_folder=dest_path.parent.name,
                page_count=doc.page_count,
            )
            self.bibtex.save()

            if progress:
                progress(1.0, desc="Complete!")

            return f"Successfully added: {doc.title}\n- {len(chunks)} chunks created\n- Citation generated"

        except Exception as e:
            return f"Error processing document: {str(e)}"

    def search_documents(
        self,
        query: str,
        n_results: int = 5,
        filter_phase: str = "",
    ) -> str:
        """
        Search the document database.

        Args:
            query: Search query
            n_results: Number of results
            filter_phase: Phase filter

        Returns:
            Search results as formatted text
        """
        if not query.strip():
            return "Please enter a search query."

        results = self.retriever.search(
            query=query,
            n_results=n_results,
            filter_phase=filter_phase if filter_phase else None,
        )

        if not results:
            return "No results found."

        output_parts = [f"Found {len(results)} results:\n"]

        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            relevance = result.get("relevance", 0)

            output_parts.append(f"\n### Result {i} (relevance: {relevance:.2f})")
            output_parts.append(f"**Document:** {metadata.get('title', 'Unknown')}")

            page = metadata.get("page_start")
            if page and page > 0:
                output_parts.append(f"**Page:** {page}")

            section = metadata.get("section")
            if section:
                output_parts.append(f"**Section:** {section}")

            output_parts.append(f"\n{result['content'][:500]}...")

            footnotes = metadata.get("footnotes", {})
            if footnotes:
                output_parts.append("\n**Footnotes:**")
                for fn_num, fn_text in list(footnotes.items())[:3]:
                    output_parts.append(f"- [{fn_num}] {fn_text[:100]}...")

        return "\n".join(output_parts)

    def get_stats(self) -> str:
        """Get database statistics."""
        stats = self.chroma_client.get_stats()
        citations = self.bibtex.list_entries()

        return f"""## Database Statistics

- **Total Documents:** {stats['total_documents']}
- **Total Chunks:** {stats['total_chunks']}
- **Citations:** {len(citations)}
- **Collection:** {stats['collection_name']}
"""

    def list_all_documents(self) -> str:
        """List all documents in the database."""
        documents = self.chroma_client.list_documents()

        if not documents:
            return "No documents in database. Upload PDFs to get started."

        docs_by_folder = {}
        for doc in documents:
            folder = doc.get("parent_folder", "Other")
            if folder not in docs_by_folder:
                docs_by_folder[folder] = []
            docs_by_folder[folder].append(doc)

        output_parts = [f"## Documents ({len(documents)} total)\n"]

        for folder in sorted(docs_by_folder.keys()):
            output_parts.append(f"\n### {folder}")
            for doc in docs_by_folder[folder]:
                output_parts.append(
                    f"- **{doc['title']}**\n  "
                    f"ID: `{doc['doc_id']}` | Chunks: {doc['total_chunks']}"
                )

        return "\n".join(output_parts)

    def list_citations(self) -> str:
        """List all citations in BibTeX format."""
        entries = self.bibtex.list_entries()

        if not entries:
            return "No citations yet."

        output_parts = ["## Citations\n"]

        for entry in entries:
            output_parts.append(
                f"- **@{entry['type']}{{{entry['key']}}}**\n"
                f"  {entry['author']} ({entry['year']}). {entry['title']}"
            )

        return "\n".join(output_parts)

    def export_bibtex(self) -> str:
        """Export all citations as BibTeX."""
        return self.bibtex.export_all()

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="Research Document Manager") as interface:
            gr.Markdown("# My Research Notebook - Document Manager")

            with gr.Tabs():
                with gr.Tab("Upload"):
                    gr.Markdown("Upload PDF documents to add to the knowledge base.")
                    file_input = gr.File(
                        label="Select PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    upload_btn = gr.Button("Upload and Process", variant="primary")
                    upload_output = gr.Textbox(
                        label="Status",
                        lines=5,
                        interactive=False,
                    )

                    upload_btn.click(
                        fn=self.upload_document,
                        inputs=[file_input],
                        outputs=[upload_output],
                    )

                with gr.Tab("Search"):
                    gr.Markdown("Search the document database.")
                    with gr.Row():
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., research topic or concept",
                        )
                        search_n = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Results",
                        )
                    folder_filter = gr.Textbox(
                        label="Filter by Folder",
                        placeholder="Enter folder name to filter",
                        value="",
                    )
                    search_btn = gr.Button("Search", variant="primary")
                    search_output = gr.Markdown(label="Results")

                    search_btn.click(
                        fn=self.search_documents,
                        inputs=[search_query, search_n, folder_filter],
                        outputs=[search_output],
                    )

                with gr.Tab("Documents"):
                    gr.Markdown("Browse all documents in the knowledge base.")
                    refresh_docs_btn = gr.Button("Refresh List")
                    docs_output = gr.Markdown()

                    refresh_docs_btn.click(
                        fn=self.list_all_documents,
                        outputs=[docs_output],
                    )
                    interface.load(
                        fn=self.list_all_documents,
                        outputs=[docs_output],
                    )

                with gr.Tab("Citations"):
                    gr.Markdown("Manage BibTeX citations.")
                    with gr.Row():
                        refresh_cite_btn = gr.Button("Refresh Citations")
                        export_btn = gr.Button("Export BibTeX")

                    citations_output = gr.Markdown()
                    bibtex_output = gr.Code(
                        label="BibTeX Export",
                        language="latex",
                        visible=False,
                    )

                    refresh_cite_btn.click(
                        fn=self.list_citations,
                        outputs=[citations_output],
                    )

                    def show_export():
                        return gr.update(visible=True, value=self.export_bibtex())

                    export_btn.click(
                        fn=show_export,
                        outputs=[bibtex_output],
                    )

                with gr.Tab("Stats"):
                    gr.Markdown("Database statistics.")
                    refresh_stats_btn = gr.Button("Refresh Stats")
                    stats_output = gr.Markdown()

                    refresh_stats_btn.click(
                        fn=self.get_stats,
                        outputs=[stats_output],
                    )
                    interface.load(
                        fn=self.get_stats,
                        outputs=[stats_output],
                    )

        return interface

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(**kwargs)


def main():
    """Entry point for the document manager UI."""
    ui = DocumentManagerUI()
    ui.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
