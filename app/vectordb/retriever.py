"""Document retriever with search and filtering capabilities."""

from typing import Optional

from .chroma_client import ChromaClient
from ..config import get_settings


class DocumentRetriever:
    """High-level retriever for searching research documents."""

    def __init__(self, chroma_client: Optional[ChromaClient] = None):
        """
        Initialize the retriever.

        Args:
            chroma_client: ChromaDB client instance
        """
        self.client = chroma_client or ChromaClient()
        self.settings = get_settings()

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_phase: Optional[str] = None,
        filter_doc_id: Optional[str] = None,
        filter_doc_ids: Optional[list[str]] = None,
        min_relevance: float = 0.0,
    ) -> list[dict]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_phase: Filter by folder name (partial match)
            filter_doc_id: Filter by specific document ID
            filter_doc_ids: Filter by multiple document IDs (used for checkbox selection)
            min_relevance: Minimum relevance score (0-1, where 1 is most similar)

        Returns:
            List of search results with content, metadata, and citations
        """
        n_results = min(n_results, self.settings.max_n_results)

        filter_dict = self._build_filter(filter_phase, filter_doc_id, filter_doc_ids)

        results = self.client.search(
            query=query,
            n_results=n_results * 2,
            filter_dict=filter_dict,
        )

        # Double-check filter_doc_ids enforcement (defense in depth)
        # ChromaDB $in filter should work, but ensure no unselected docs slip through
        if filter_doc_ids is not None:
            results = [
                r for r in results
                if r["metadata"].get("doc_id") in filter_doc_ids
            ]

        if min_relevance > 0:
            results = [
                r for r in results
                if self._distance_to_relevance(r["distance"]) >= min_relevance
            ]

        results = results[:n_results]

        for result in results:
            result["relevance"] = self._distance_to_relevance(result["distance"])
            result["citation"] = self._format_citation(result)

        return results

    def _build_filter(
        self,
        filter_phase: Optional[str],
        filter_doc_id: Optional[str],
        filter_doc_ids: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Build ChromaDB filter from parameters."""
        conditions = []

        if filter_phase:
            conditions.append({
                "parent_folder": {"$contains": filter_phase}
            })

        if filter_doc_id:
            conditions.append({
                "doc_id": {"$eq": filter_doc_id}
            })
        elif filter_doc_ids:
            conditions.append({
                "doc_id": {"$in": filter_doc_ids}
            })

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def _distance_to_relevance(self, distance: float) -> float:
        """Convert distance to relevance score (0-1)."""
        return max(0, 1 - (distance / 2))

    def _format_citation(self, result: dict) -> str:
        """Format a citation string for the result."""
        metadata = result["metadata"]
        title = metadata.get("title", "Unknown")
        page = metadata.get("page_start")

        if page and page > 0:
            return f"{title}, p. {page}"
        return title

    def get_context_for_query(
        self,
        query: str,
        n_results: int = 5,
        include_footnotes: bool = True,
    ) -> str:
        """
        Get formatted context for a query.

        Args:
            query: Search query
            n_results: Number of results
            include_footnotes: Whether to include footnote text

        Returns:
            Formatted context string
        """
        results = self.search(query, n_results=n_results)

        context_parts = []
        for i, result in enumerate(results, 1):
            citation = result["citation"]
            content = result["content"]

            part = f"[Source {i}: {citation}]\n{content}"

            if include_footnotes:
                footnotes = result["metadata"].get("footnotes", {})
                if footnotes:
                    fn_text = "\n".join(
                        f"  [{k}] {v}" for k, v in footnotes.items()
                    )
                    part += f"\n\nFootnotes:\n{fn_text}"

            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)

    def search_by_section(
        self,
        section_keywords: list[str],
        n_results: int = 10,
    ) -> list[dict]:
        """
        Search for chunks from specific sections.

        Args:
            section_keywords: Keywords to match in section titles
            n_results: Number of results

        Returns:
            List of matching chunks
        """
        query = " ".join(section_keywords)
        results = self.client.search(query, n_results=n_results * 2)

        filtered = []
        for result in results:
            section = result["metadata"].get("section", "").lower()
            if any(kw.lower() in section for kw in section_keywords):
                filtered.append(result)

        return filtered[:n_results]

    def get_document_summary(self, doc_id: str) -> dict:
        """
        Get a summary of a specific document.

        Args:
            doc_id: Document ID

        Returns:
            Document summary with metadata and sample content
        """
        chunks = self.client.get_document_chunks(doc_id)

        if not chunks:
            return {"error": f"Document {doc_id} not found"}

        first_chunk = chunks[0]
        metadata = first_chunk["metadata"]

        return {
            "doc_id": doc_id,
            "title": metadata.get("title", "Unknown"),
            "source_path": metadata.get("source_path", ""),
            "total_chunks": len(chunks),
            "total_tokens": sum(c["metadata"].get("token_count", 0) for c in chunks),
            "sections": list(set(
                c["metadata"].get("section", "") for c in chunks if c["metadata"].get("section")
            )),
            "sample_content": first_chunk["content"][:500] + "...",
        }

    def find_related_chunks(
        self,
        chunk_id: str,
        n_results: int = 3,
    ) -> list[dict]:
        """
        Find chunks related to a specific chunk.

        Args:
            chunk_id: Source chunk ID
            n_results: Number of related chunks

        Returns:
            List of related chunks from other documents
        """
        source_results = self.client.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )

        if not source_results["documents"]:
            return []

        source_content = source_results["documents"][0]
        source_doc_id = source_results["metadatas"][0].get("doc_id")

        results = self.client.search(source_content, n_results=n_results + 5)

        related = [
            r for r in results
            if r["metadata"].get("doc_id") != source_doc_id
        ]

        return related[:n_results]

    def get_by_page(
        self,
        doc_title_query: str,
        page_number: Optional[int] = None,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Get document content by title match and optional page number.

        This enables exact retrieval like "fetch Prel. Doc. No 3B page 3".

        Args:
            doc_title_query: Partial document title to match (case-insensitive)
            page_number: Specific page number to retrieve (optional)
            filter_doc_ids: Only search within these doc_ids (for source filtering)

        Returns:
            List of chunks matching the criteria, sorted by page/chunk order
        """
        # Get all documents to find matching titles
        documents = self.client.list_documents()

        # Find documents with matching titles
        doc_title_lower = doc_title_query.lower()
        matching_doc_ids = []

        for doc in documents:
            title = doc.get("title", "").lower()
            doc_id = doc.get("doc_id", "")

            # Check if title matches query
            if doc_title_lower in title:
                # If filter_doc_ids is set, only include if in allowed list
                if filter_doc_ids is None or doc_id in filter_doc_ids:
                    matching_doc_ids.append(doc_id)

        if not matching_doc_ids:
            return []

        # Get chunks from matching documents
        all_chunks = []
        for doc_id in matching_doc_ids:
            chunks = self.client.get_document_chunks(doc_id)
            all_chunks.extend(chunks)

        # Filter by page number if specified
        if page_number is not None:
            all_chunks = [
                c for c in all_chunks
                if c["metadata"].get("page_start") == page_number
            ]

        # Sort by doc_id, then page, then chunk_index
        all_chunks.sort(key=lambda x: (
            x["metadata"].get("doc_id", ""),
            x["metadata"].get("page_start", 0),
            x["metadata"].get("chunk_index", 0),
        ))

        return all_chunks

    def search_by_title(
        self,
        title_query: str,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Find documents whose titles match a query string.

        Args:
            title_query: Partial title to search for
            filter_doc_ids: Only include documents in this list

        Returns:
            List of matching document metadata
        """
        documents = self.client.list_documents()
        title_lower = title_query.lower()

        matches = []
        for doc in documents:
            title = doc.get("title", "").lower()
            doc_id = doc.get("doc_id", "")

            if title_lower in title:
                if filter_doc_ids is None or doc_id in filter_doc_ids:
                    matches.append(doc)

        return matches
