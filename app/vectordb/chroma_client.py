"""ChromaDB client wrapper with persistence."""

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import get_settings
from ..parsers.simple_chunker import SimpleChunk
from .embeddings import OpenAIEmbeddings

# Average characters per token for English text (OpenAI tokenizer approximation)
CHARS_PER_TOKEN_ESTIMATE = 4


class ChromaClient:
    """ChromaDB client with persistent storage and OpenAI embeddings."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the ChromaDB client.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
        """
        settings = get_settings()
        self.persist_directory = persist_directory or settings.chroma_db_dir
        self.collection_name = collection_name or settings.collection_name

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.embeddings = OpenAIEmbeddings()
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get or create the document collection."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Research notebook documents"},
        )

    def add_simple_chunks(self, chunks: list[SimpleChunk], show_progress: bool = True):
        """
        Add simple chunks to the collection.

        Args:
            chunks: List of SimpleChunk objects
            show_progress: Whether to show progress
        """
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [self._simple_chunk_to_metadata(chunk) for chunk in chunks]

        if show_progress:
            print(f"Generating embeddings for {len(chunks)} chunks...")

        embeddings = self.embeddings.embed_texts(documents)

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                embeddings=embeddings[i:end_idx],
            )

            if show_progress:
                print(f"  Added chunks {i+1}-{end_idx} of {len(chunks)}")

    def _simple_chunk_to_metadata(self, chunk: SimpleChunk) -> dict:
        """Convert simple chunk metadata to ChromaDB-compatible format."""
        return {
            "doc_id": chunk.doc_id,
            "source_path": chunk.source_path,
            "title": chunk.title,
            "page_start": chunk.page_number,
            "page_end": chunk.page_number,
            "section": "",
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "token_count": chunk.char_count // CHARS_PER_TOKEN_ESTIMATE,
            "footnotes": "{}",
            "parser": chunk.metadata.get("parser", "simple"),
            "parent_folder": chunk.parent_folder,
        }

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar documents.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of search results with content and metadata
        """
        query_embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            metadata["footnotes"] = json.loads(metadata.get("footnotes", "{}"))

            formatted_results.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": metadata,
            })

        return formatted_results

    def get_document_chunks(self, doc_id: str) -> list[dict]:
        """
        Get all chunks for a specific document.

        Args:
            doc_id: Document ID

        Returns:
            List of chunks for the document
        """
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )

        chunks = []
        for i in range(len(results["ids"])):
            metadata = results["metadatas"][i]
            metadata["footnotes"] = json.loads(metadata.get("footnotes", "{}"))

            chunks.append({
                "chunk_id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": metadata,
            })

        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        return chunks

    def list_documents(self) -> list[dict]:
        """
        List all unique documents in the collection.

        Returns:
            List of document metadata
        """
        results = self.collection.get(include=["metadatas"])

        documents = {}
        for metadata in results["metadatas"]:
            doc_id = metadata.get("doc_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "doc_id": doc_id,
                    "title": metadata.get("title", "Unknown"),
                    "source_path": metadata.get("source_path", ""),
                    "total_chunks": metadata.get("total_chunks", 0),
                    "parent_folder": metadata.get("parent_folder", ""),
                }

        return list(documents.values())

    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            doc_id: Document ID

        Returns:
            Number of chunks deleted
        """
        results = self.collection.get(where={"doc_id": doc_id})
        chunk_ids = results["ids"]

        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

        return len(chunk_ids)

    def get_stats(self) -> dict:
        """
        Get collection statistics.

        Returns:
            Dictionary with collection stats
        """
        count = self.collection.count()
        documents = self.list_documents()

        return {
            "total_chunks": count,
            "total_documents": len(documents),
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }

    def clear(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()
