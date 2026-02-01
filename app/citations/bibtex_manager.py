"""BibTeX citation manager for research documents."""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pybtex.database import BibliographyData, Entry

from ..config import get_settings


class BibTeXManager:
    """Manages BibTeX citations for research source documents."""

    def __init__(self, bib_path: Optional[Path] = None):
        """
        Initialize the BibTeX manager.

        Args:
            bib_path: Path to the master .bib file
        """
        settings = get_settings()
        self.bib_path = bib_path or settings.citations_dir / "references.bib"
        self.bib_path.parent.mkdir(parents=True, exist_ok=True)

        self.bib_data = self._load_or_create()

    def _load_or_create(self) -> BibliographyData:
        """Load existing bib file or create new one."""
        if self.bib_path.exists():
            try:
                from pybtex.database import parse_file
                return parse_file(str(self.bib_path))
            except Exception:
                pass
        return BibliographyData()

    def generate_bibtex_key(self, title: str, year: Optional[str] = None) -> str:
        """
        Generate a BibTeX key from document title.

        Args:
            title: Document title
            year: Publication year

        Returns:
            BibTeX key (e.g., "smith2024research")
        """
        words = re.sub(r"[^\w\s]", "", title.lower()).split()

        author_words = []
        for word in words[:3]:
            if word not in {"the", "a", "an", "of", "for", "and", "in", "on"}:
                author_words.append(word)
            if len(author_words) >= 2:
                break

        key_base = "".join(author_words) if author_words else "doc"

        if not year:
            year_match = re.search(r"\b(19|20)\d{2}\b", title)
            year = year_match.group() if year_match else str(datetime.now().year)

        key = f"{key_base}{year}"

        counter = 1
        original_key = key
        while key in self.bib_data.entries:
            key = f"{original_key}{chr(ord('a') + counter - 1)}"
            counter += 1

        return key

    def create_entry_from_metadata(
        self,
        doc_id: str,
        title: str,
        source_path: str,
        parent_folder: str = "",
        page_count: int = 0,
    ) -> Entry:
        """
        Create a BibTeX entry from document metadata.

        Args:
            doc_id: Document ID
            title: Document title
            source_path: Path to source file
            parent_folder: Parent folder name (contains meeting info)
            page_count: Number of pages

        Returns:
            BibTeX Entry object
        """
        year = self._extract_year(title, parent_folder)
        author = self._extract_author(title)
        entry_type = self._determine_entry_type(title)

        fields = {
            "title": title,
            "author": author,
            "year": year,
            "note": f"Document ID: {doc_id}",
            "file": source_path,
        }

        if page_count > 0:
            fields["pages"] = str(page_count)

        meeting = self._extract_meeting(parent_folder)
        if meeting:
            fields["howpublished"] = meeting

        return Entry(entry_type, fields)

    def _extract_year(self, title: str, parent_folder: str) -> str:
        """Extract publication year from title or folder name."""
        for text in [title, parent_folder]:
            year_match = re.search(r"\b(19|20)\d{2}\b", text)
            if year_match:
                return year_match.group()
        return str(datetime.now().year)

    def _extract_author(self, title: str) -> str:
        """Extract author from title using bracket notation."""
        # Look for bracket pattern: [DocRef, Author, ...]
        bracket_match = re.match(r"\[([^\]]+)\]", title)
        if bracket_match:
            content = bracket_match.group(1)
            parts = content.split(",")
            if len(parts) > 1:
                return parts[1].strip()

        return "Unknown Author"

    def _determine_entry_type(self, title: str) -> str:
        """Determine BibTeX entry type from title."""
        title_lower = title.lower()

        if "report" in title_lower:
            return "techreport"
        if "paper" in title_lower or "working paper" in title_lower:
            return "article"
        if "presentation" in title_lower or "slides" in title_lower:
            return "misc"
        if "note" in title_lower:
            return "misc"

        return "techreport"

    def _extract_meeting(self, parent_folder: str) -> str:
        """Extract meeting/event information from folder name."""
        meeting_match = re.search(
            r"(\d+)(?:st|nd|rd|th)\s+(?:meeting|session)[^(]*\(([^)]+)\)",
            parent_folder,
            re.IGNORECASE,
        )
        if meeting_match:
            num = meeting_match.group(1)
            date = meeting_match.group(2)
            ordinal = "st" if num == "1" else "nd" if num == "2" else "rd" if num == "3" else "th"
            return f"{num}{ordinal} Meeting ({date})"

        return ""

    def add_document(
        self,
        doc_id: str,
        title: str,
        source_path: str,
        parent_folder: str = "",
        page_count: int = 0,
    ) -> str:
        """
        Add a document to the bibliography.

        Args:
            doc_id: Document ID
            title: Document title
            source_path: Path to source file
            parent_folder: Parent folder name
            page_count: Number of pages

        Returns:
            BibTeX key for the document
        """
        key = self.generate_bibtex_key(title)
        entry = self.create_entry_from_metadata(
            doc_id, title, source_path, parent_folder, page_count
        )

        self.bib_data.entries[key] = entry
        return key

    def get_citation(self, key: str) -> Optional[str]:
        """
        Get formatted citation for a BibTeX key.

        Args:
            key: BibTeX key

        Returns:
            Formatted citation string, or None if not found
        """
        if key not in self.bib_data.entries:
            return None

        entry = self.bib_data.entries[key]
        fields = entry.fields

        author = fields.get("author", "Unknown")
        year = fields.get("year", "n.d.")
        title = fields.get("title", "Untitled")

        return f"{author} ({year}). {title}"

    def get_bibtex_entry(self, key: str) -> Optional[str]:
        """
        Get raw BibTeX entry string.

        Args:
            key: BibTeX key

        Returns:
            BibTeX entry string, or None if not found
        """
        if key not in self.bib_data.entries:
            return None

        entry = self.bib_data.entries[key]
        lines = [f"@{entry.type}{{{key},"]

        for field, value in entry.fields.items():
            lines.append(f"  {field} = {{{value}}},")

        lines.append("}")
        return "\n".join(lines)

    def find_by_doc_id(self, doc_id: str) -> Optional[str]:
        """
        Find BibTeX key by document ID.

        Args:
            doc_id: Document ID

        Returns:
            BibTeX key, or None if not found
        """
        for key, entry in self.bib_data.entries.items():
            note = entry.fields.get("note", "")
            if f"Document ID: {doc_id}" in note:
                return key
        return None

    def save(self) -> None:
        """Save bibliography to file."""
        self.bib_data.to_file(str(self.bib_path))

    def export_all(self) -> str:
        """
        Export all entries as BibTeX string.

        Returns:
            Complete BibTeX file contents
        """
        lines = []
        for key in self.bib_data.entries:
            entry_str = self.get_bibtex_entry(key)
            if entry_str:
                lines.append(entry_str)
        return "\n\n".join(lines)

    def list_entries(self) -> list[dict]:
        """
        List all bibliography entries.

        Returns:
            List of entry summaries
        """
        entries = []
        for key, entry in self.bib_data.entries.items():
            entries.append({
                "key": key,
                "title": entry.fields.get("title", "Untitled"),
                "author": entry.fields.get("author", "Unknown"),
                "year": entry.fields.get("year", "n.d."),
                "type": entry.type,
            })
        return entries
