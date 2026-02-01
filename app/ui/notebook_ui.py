"""NotebookLM-style UI with sources panel, chat, and model selection."""

import os
from pathlib import Path
from typing import Optional
import shutil

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from openai import OpenAI
from anthropic import Anthropic

from ..config import get_settings
from ..parsers import MultiFormatParser, SimpleChunker
from ..vectordb import ChromaClient, DocumentRetriever
from ..citations import BibTeXManager

# Optional imports for additional providers
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


# Model configurations
# Cost format: ($input/$output) per 1M tokens, + indicates additional thinking/reasoning token costs
MODELS = {
    # ===== Anthropic =====
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "name": "Claude Sonnet 4 ($3/$15)",
        "supports_thinking": False,
    },
    "claude-sonnet-4-thinking": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "name": "Claude Sonnet 4 Thinking ($3/$15+)",
        "supports_thinking": True,
        "thinking_budget": 10000,
    },
    "claude-opus-4": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-20250514",
        "name": "Claude Opus 4 ($15/$75)",
        "supports_thinking": False,
    },
    "claude-opus-4-thinking": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-20250514",
        "name": "Claude Opus 4 Thinking ($15/$75+)",
        "supports_thinking": True,
        "thinking_budget": 20000,
    },
    "claude-3.7-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-7-sonnet-20250219",
        "name": "Claude 3.7 Sonnet ($3/$15)",
        "supports_thinking": False,
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "name": "Claude 3.5 Sonnet ($3/$15)",
        "supports_thinking": False,
    },
    # ===== OpenAI =====
    "gpt-5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2",
        "name": "GPT-5.2 ($5/$15)",
    },
    "gpt-5.1": {
        "provider": "openai",
        "model_id": "gpt-5.1",
        "name": "GPT-5.1 ($5/$15)",
    },
    "gpt-5": {
        "provider": "openai",
        "model_id": "gpt-5",
        "name": "GPT-5 ($5/$15)",
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "name": "GPT-4o ($2.5/$10)",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "name": "GPT-4o Mini ($0.15/$0.6)",
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "model_id": "gpt-4-turbo",
        "name": "GPT-4 Turbo ($10/$30)",
    },
    # OpenAI reasoning models
    "o3": {
        "provider": "openai",
        "model_id": "o3",
        "name": "o3 Reasoning ($10/$40+)",
    },
    "o3-pro": {
        "provider": "openai",
        "model_id": "o3-pro",
        "name": "o3 Pro ($20/$80+)",
    },
    "o3-mini": {
        "provider": "openai",
        "model_id": "o3-mini",
        "name": "o3 Mini ($1.1/$4.4)",
    },
    "o1": {
        "provider": "openai",
        "model_id": "o1",
        "name": "o1 Reasoning ($15/$60+)",
    },
    "o1-pro": {
        "provider": "openai",
        "model_id": "o1-pro",
        "name": "o1 Pro ($150/$600+)",
    },
    "o1-mini": {
        "provider": "openai",
        "model_id": "o1-mini",
        "name": "o1 Mini ($1.1/$4.4)",
    },
    # ===== Google Gemini =====
    "gemini-2.0-flash": {
        "provider": "google",
        "model_id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash (Free tier)",
    },
    "gemini-2.0-pro": {
        "provider": "google",
        "model_id": "gemini-2.0-pro-exp",
        "name": "Gemini 2.0 Pro ($1.25/$5)",
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "model_id": "gemini-1.5-pro",
        "name": "Gemini 1.5 Pro ($1.25/$5)",
    },
    # ===== DeepSeek =====
    "deepseek-r1": {
        "provider": "deepseek",
        "model_id": "deepseek-reasoner",
        "name": "DeepSeek R1 ($0.55/$2.19)",
    },
    "deepseek-v3": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "name": "DeepSeek V3 ($0.27/$1.10)",
    },
    # ===== Groq (Free) =====
    "groq-llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B [Groq Free]",
    },
    "groq-mixtral-8x7b": {
        "provider": "groq",
        "model_id": "mixtral-8x7b-32768",
        "name": "Mixtral 8x7B [Groq Free]",
    },
    "groq-gemma2-9b": {
        "provider": "groq",
        "model_id": "gemma2-9b-it",
        "name": "Gemma 2 9B [Groq Free]",
    },
    # ===== xAI Grok =====
    "grok-2": {
        "provider": "grok",
        "model_id": "grok-2-latest",
        "name": "Grok 2 [xAI Free tier]",
    },
    "grok-2-vision": {
        "provider": "grok",
        "model_id": "grok-2-vision-1212",
        "name": "Grok 2 Vision [xAI Free tier]",
    },
    # ===== OpenRouter (Free Models) =====
    "openrouter-llama-3.3-70b": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct:free",
        "name": "Llama 3.3 70B [OpenRouter Free]",
    },
    "openrouter-gemma-2-9b": {
        "provider": "openrouter",
        "model_id": "google/gemma-2-9b-it:free",
        "name": "Gemma 2 9B [OpenRouter Free]",
    },
    "openrouter-mistral-7b": {
        "provider": "openrouter",
        "model_id": "mistralai/mistral-7b-instruct:free",
        "name": "Mistral 7B [OpenRouter Free]",
    },
}


class NotebookUI:
    """NotebookLM-style interface with sources panel and chat."""

    DEFAULT_SYSTEM_PROMPT = """You are an expert research assistant helping draft documents based on your knowledge base.
You have access to source documents uploaded by the user.

Instructions:
- ONLY answer based on the provided context. Do NOT use any external knowledge.
- If the requested information is not in the provided context, say: "This information is not available in the selected sources."
- When citing sources, use the filename: [Filename, Page Y]
- Example: [Prel. Doc. No 3B, Page 17]
- Quote exact text with quotation marks when referencing specific passages
- Be precise and use legal/technical terminology appropriately
- Format responses with clear structure (headings, bullets) when helpful

For exact quotes:
- Provide verbatim text from the Knowledge Base Context
- Always include DocID and page number
- The context above contains the full searchable text

IMPORTANT: You must NEVER provide information that is not explicitly present in the context provided above."""

    def __init__(self):
        """Initialize UI components."""
        self.settings = get_settings()
        self.settings.ensure_directories()

        self.chroma_client = ChromaClient()
        self.retriever = DocumentRetriever(self.chroma_client)
        self.bibtex = BibTeXManager()
        self.parser = MultiFormatParser()
        self.chunker = SimpleChunker(chunk_size=2000, chunk_overlap=200)

        # Initialize API clients (validate keys are not empty)
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_client = OpenAI() if openai_key else None

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.anthropic_client = Anthropic() if anthropic_key else None

        # DeepSeek client (OpenAI-compatible)
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.deepseek_client = OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com"
        ) if deepseek_key else None

        # Groq client
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_client = None
        if GROQ_AVAILABLE and groq_key:
            self.groq_client = Groq()

        # Google Gemini
        google_key = os.getenv("GOOGLE_API_KEY", "").strip()
        self.google_client = None
        if GOOGLE_AVAILABLE and google_key:
            self.google_client = genai.Client(api_key=google_key)

        # xAI Grok client (OpenAI-compatible)
        grok_key = os.getenv("GROK_API_KEY", "").strip()
        self.grok_client = OpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
        ) if grok_key else None

        # OpenRouter client (OpenAI-compatible)
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openrouter_client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        ) if openrouter_key else None

        self.temp_context = ""  # Temporary file context for chat
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # Get .env file path
        self.env_path = Path(__file__).parent.parent.parent / ".env"

    # ===== Sources Panel Functions =====

    def get_sources_list(self) -> list[list]:
        """Get list of sources for the dataframe."""
        documents = self.chroma_client.list_documents()

        if not documents:
            return []

        rows = []
        for doc in documents:
            rows.append([
                doc.get("title", "Unknown")[:50],
                doc.get("parent_folder", "")[:20],
                doc.get("total_chunks", 0),
                doc.get("doc_id", ""),
            ])

        return rows

    def get_sources_choices(self) -> list[tuple[str, str]]:
        """Get sources as (label, doc_id) tuples for CheckboxGroup."""
        documents = self.chroma_client.list_documents()
        if not documents:
            return []

        choices = []
        for doc in documents:
            title = doc.get("title", "Unknown")[:40]
            phase = doc.get("parent_folder", "")[:15]
            chunks = doc.get("total_chunks", 0)
            doc_id = doc.get("doc_id", "")
            label = f"{title} | {phase} ({chunks} chunks)"
            choices.append((label, doc_id))

        return choices

    def get_all_doc_ids(self) -> list[str]:
        """Get all document IDs."""
        documents = self.chroma_client.list_documents()
        return [doc.get("doc_id", "") for doc in documents if doc.get("doc_id")]

    def delete_selected_sources(self, selected_doc_ids: list[str]) -> tuple[str, list[tuple[str, str]], str]:
        """Delete multiple selected sources."""
        if not selected_doc_ids:
            choices = self.get_sources_choices()
            return "No sources selected.", choices, self.get_stats_text()

        # Cache document list once for efficiency
        documents = self.chroma_client.list_documents()
        doc_map = {d.get("doc_id"): d for d in documents}

        deleted = []
        errors = []

        for doc_id in selected_doc_ids:
            try:
                doc_info = doc_map.get(doc_id)
                if doc_info:
                    self.chroma_client.delete_document(doc_id)
                    deleted.append(doc_info.get("title", doc_id)[:30])
            except Exception as e:
                errors.append(f"{doc_id}: {str(e)}")

        choices = self.get_sources_choices()
        result_parts = []
        if deleted:
            result_parts.append(f"[DELETED] {len(deleted)} sources")
        if errors:
            result_parts.append(f"[ERROR] {len(errors)} failures")

        return " | ".join(result_parts) or "No changes", choices, self.get_stats_text()

    def add_source(self, file_paths: list[str] | str | None, progress: gr.Progress = None) -> tuple[str, list]:
        """Add document(s) to the vector database."""
        if not file_paths:
            return "No file selected.", self.get_sources_list()

        # Normalize to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        supported = {".pdf", ".md", ".txt", ".xml", ".docx"}
        results = []

        for i, fp in enumerate(file_paths):
            file_path = Path(fp)

            if file_path.suffix.lower() not in supported:
                results.append(f"❌ {file_path.name}: Unsupported format")
                continue

            try:
                # Copy to sources directory
                dest_path = self.settings.sources_dir / file_path.name
                shutil.copy(file_path, dest_path)

                if progress:
                    progress((i + 0.3) / len(file_paths), desc=f"Parsing {file_path.name}...")

                # Parse
                page_chunks = self.parser.parse(dest_path)
                if not page_chunks:
                    results.append(f"❌ {file_path.name}: Failed to parse")
                    continue

                if progress:
                    progress((i + 0.6) / len(file_paths), desc=f"Chunking {file_path.name}...")

                # Chunk
                chunks = self.chunker.chunk_pages(page_chunks)
                if not chunks:
                    results.append(f"❌ {file_path.name}: No content extracted")
                    continue

                if progress:
                    progress((i + 0.9) / len(file_paths), desc=f"Adding {file_path.name}...")

                # Add to vector DB
                self.chroma_client.add_simple_chunks(chunks, show_progress=False)

                # Add citation
                doc_id = chunks[0].doc_id
                title = chunks[0].title
                total_pages = page_chunks[0].metadata.get("total_pages", len(page_chunks))

                self.bibtex.add_document(
                    doc_id=doc_id,
                    title=title,
                    source_path=str(dest_path),
                    parent_folder=dest_path.parent.name,
                    page_count=total_pages,
                )
                self.bibtex.save()

                results.append(f"✅ {title} ({len(chunks)} chunks)")

            except Exception as e:
                results.append(f"❌ {file_path.name}: {str(e)}")

        return "\n".join(results), self.get_sources_list()

    def delete_source(self, doc_id: str) -> tuple[str, list]:
        """Delete a source from the vector database."""
        if not doc_id:
            return "No document selected.", self.get_sources_list()

        try:
            documents = self.chroma_client.list_documents()
            doc_info = next((d for d in documents if d.get("doc_id") == doc_id), None)

            if not doc_info:
                return f"Document not found: {doc_id}", self.get_sources_list()

            self.chroma_client.delete_document(doc_id)

            return f"🗑️ Deleted: {doc_info.get('title', doc_id)}", self.get_sources_list()

        except Exception as e:
            return f"❌ Error deleting: {str(e)}", self.get_sources_list()

    def get_stats_text(self) -> str:
        """Get database statistics as text."""
        stats = self.chroma_client.get_stats()
        return f"📊 Documents: {stats['total_documents']} | Chunks: {stats['total_chunks']}"

    def update_system_prompt(self, prompt: str) -> str:
        """Update the system prompt."""
        self.system_prompt = prompt.strip() if prompt.strip() else self.DEFAULT_SYSTEM_PROMPT
        return "✅ System prompt updated"

    def load_system_prompt_file(self, file_path: str) -> tuple[str, str]:
        """Load system prompt from file."""
        if not file_path:
            return self.system_prompt, "⚠️ No file selected"

        path = Path(file_path)
        if path.suffix.lower() not in {".md", ".txt"}:
            return self.system_prompt, "❌ Only .md or .txt files supported"

        try:
            content = path.read_text(encoding="utf-8")
            self.system_prompt = content.strip()
            return self.system_prompt, f"✅ Loaded: {path.name}"
        except Exception as e:
            return self.system_prompt, f"❌ Error: {str(e)}"

    # ===== Chat Functions =====

    def upload_temp_file(self, file_paths: list[str] | str | None) -> str:
        """Upload file(s) for temporary chat context."""
        if not file_paths:
            return "📎 No temporary file loaded"

        # Normalize to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        supported = {".pdf", ".md", ".txt", ".xml", ".docx"}
        all_text_parts = []
        loaded_files = []

        for fp in file_paths:
            file_path = Path(fp)

            if file_path.suffix.lower() not in supported:
                continue

            try:
                page_chunks = self.parser.parse(file_path)
                if not page_chunks:
                    continue

                text_parts = []
                for chunk in page_chunks:
                    text_parts.append(f"[Section {chunk.page_number}]\n{chunk.content}")

                all_text_parts.append(f"=== {file_path.name} ===\n" + "\n\n".join(text_parts))
                loaded_files.append(file_path.name)

            except Exception:
                continue

        if not loaded_files:
            return "❌ Failed to load any files"

        self.temp_context = "\n\n".join(all_text_parts)
        return f"📄 Loaded: {', '.join(loaded_files)} ({len(loaded_files)} files)"

    def clear_temp_context(self) -> str:
        """Clear temporary file context."""
        self.temp_context = ""
        return "📎 No temporary file loaded"

    def _try_page_fetch(
        self,
        message: str,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> Optional[list[dict]]:
        """
        Try to detect and handle page/document fetch requests.

        Detects patterns like:
        - "fetch Prel. Doc. No 3B page 3"
        - "get page 5 from document X"
        - "Prel. Doc. No 3B 109번 문단"

        Args:
            message: User message
            filter_doc_ids: Only allow fetching from these doc_ids

        Returns:
            List of chunks if fetch pattern detected, None otherwise
        """
        import re

        msg_lower = message.lower()

        # Detect fetch/get patterns
        fetch_keywords = ["fetch", "get", "retrieve", "show", "가져와", "보여줘", "찾아"]
        has_fetch = any(kw in msg_lower for kw in fetch_keywords)

        # Detect document reference patterns
        doc_patterns = [
            r"(?:prel\.?\s*)?doc\.?\s*(?:no\.?\s*)?(\d+[a-zA-Z]*)",  # Prel. Doc. No 3B
            r"document\s+([^\s,]+)",  # document X
            r"문서\s+([^\s,]+)",  # 문서 X (Korean)
        ]

        # Detect page/paragraph patterns
        page_patterns = [
            r"page\s*(\d+)",  # page 3
            r"p\.?\s*(\d+)",  # p.3 or p 3
            r"페이지\s*(\d+)",  # 페이지 3 (Korean)
            r"(\d+)\s*페이지",  # 3페이지 (Korean)
            r"para(?:graph)?\s*(\d+)",  # para 109, paragraph 109
            r"(\d+)\s*(?:번\s*)?(?:문단|단락)",  # 109번 문단 (Korean)
        ]

        doc_match = None
        page_num = None

        # Find document reference
        for pattern in doc_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                doc_match = match.group(1) if match.groups() else match.group(0)
                break

        # If no doc pattern, try to extract from the whole message
        if not doc_match and has_fetch:
            # Try to find document title keywords like "Prel. Doc" etc.
            prel_match = re.search(r"prel\.?\s*doc\.?\s*(?:no\.?\s*)?([^\s,]+)", message, re.IGNORECASE)
            if prel_match:
                doc_match = f"Prel. Doc. No {prel_match.group(1)}"

        # Find page/paragraph number
        for pattern in page_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                page_num = int(match.group(1))
                break

        # Only proceed if we have a document reference and either page number or fetch keyword
        if doc_match and (page_num is not None or has_fetch):
            results = self.retriever.get_by_page(
                doc_title_query=doc_match,
                page_number=page_num,
                filter_doc_ids=filter_doc_ids,
            )
            if results:
                return results[:10]  # Limit to 10 chunks

        return None

    def chat_stream(
        self,
        message: str,
        history: list,
        model_key: str,
        use_sources: bool = True,
        selected_sources: Optional[list[str]] = None,
    ):
        """Process chat message with streaming response."""
        if not message.strip():
            yield history, ""
            return

        model_config = MODELS.get(model_key, MODELS["gpt-4o"])
        provider = model_config["provider"]

        # Check if provider is available
        client_map = {
            "openai": (self.openai_client, "OPENAI_API_KEY"),
            "anthropic": (self.anthropic_client, "ANTHROPIC_API_KEY"),
            "deepseek": (self.deepseek_client, "DEEPSEEK_API_KEY"),
            "groq": (self.groq_client, "GROQ_API_KEY"),
            "google": (self.google_client, "GOOGLE_API_KEY"),
            "grok": (self.grok_client, "GROK_API_KEY"),
            "openrouter": (self.openrouter_client, "OPENROUTER_API_KEY"),
        }

        client, key_name = client_map.get(provider, (None, ""))
        if not client:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ {provider.title()} API key not configured. Add {key_name} to .env"})
            yield history, ""
            return

        # Build context
        context_parts = []

        if self.temp_context:
            context_parts.append("=== Uploaded File Context ===")
            context_parts.append(self.temp_context[:8000])

        if use_sources:
            # Empty list means user explicitly deselected all - don't search
            if selected_sources is not None and len(selected_sources) == 0:
                results = []  # Skip KB search when no sources selected
            else:
                filter_doc_ids = selected_sources if selected_sources else None

                # Try page/document fetch first (e.g., "fetch Prel. Doc. No 3B page 3")
                page_results = self._try_page_fetch(message, filter_doc_ids)

                if page_results:
                    results = page_results
                else:
                    results = self.retriever.search(
                        query=message,
                        n_results=5,
                        filter_doc_ids=filter_doc_ids,
                    )
            if results:
                context_parts.append("\n=== Knowledge Base Context ===")
                for i, result in enumerate(results, 1):
                    metadata = result["metadata"]
                    source_path = metadata.get("source_path", "")
                    filename = Path(source_path).stem if source_path else metadata.get("title", "Unknown")
                    context_parts.append(
                        f"\n[Source {i}: {filename}]\n"
                        f"Page: {metadata.get('page_start', '?')}\n"
                        f"---\n"
                        f"{result['content'][:1500]}"
                    )

        context = "\n".join(context_parts)

        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {message}"
        else:
            user_content = message

        # Add user message to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        yield history, ""

        try:
            # Select streaming method
            stream_methods = {
                "anthropic": self._chat_anthropic_stream,
                "google": self._chat_google_stream,
                "deepseek": self._chat_deepseek_stream,
                "groq": self._chat_groq_stream,
                "grok": self._chat_grok_stream,
                "openrouter": self._chat_openrouter_stream,
                "openai": self._chat_openai_stream,
            }

            stream_method = stream_methods.get(provider, self._chat_openai_stream)
            full_response = ""

            for chunk in stream_method(user_content, history[:-2], model_config):
                full_response += chunk
                history[-1]["content"] = full_response
                yield history, ""

        except Exception as e:
            history[-1]["content"] = f"❌ Error: {str(e)}"
            yield history, ""

    # ===== Streaming Chat Methods =====

    def _chat_openai_stream(self, user_content: str, history: list, model_config: dict):
        """Stream chat using OpenAI API."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for h in history:
            if h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": user_content})

        stream = self.openai_client.chat.completions.create(
            model=model_config["model_id"],
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _chat_anthropic_stream(self, user_content: str, history: list, model_config: dict):
        """Stream chat using Anthropic API."""
        messages = []

        for h in history:
            if h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": user_content})

        # Check if thinking mode is enabled
        if model_config.get("supports_thinking"):
            with self.anthropic_client.messages.stream(
                model=model_config["model_id"],
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": model_config.get("thinking_budget", 10000),
                },
                system=self.system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        else:
            with self.anthropic_client.messages.stream(
                model=model_config["model_id"],
                max_tokens=4000,
                system=self.system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text

    def _chat_google_stream(self, user_content: str, history: list, model_config: dict):
        """Stream chat using Google Gemini API."""
        if not self.google_client:
            raise ValueError("Google GenAI client not initialized")

        # Build conversation contents
        contents = []
        for h in history:
            if h.get("content"):
                role = "model" if h["role"] == "assistant" else h["role"]
                contents.append(genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(h["content"])]
                ))

        contents.append(genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(user_content)]
        ))

        response = self.google_client.models.generate_content_stream(
            model=model_config["model_id"],
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.3,
                max_output_tokens=4000,
            ),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _chat_deepseek_stream(self, user_content: str, history: list, model_config: dict):
        """Stream chat using DeepSeek API (OpenAI-compatible)."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for h in history:
            if h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": user_content})

        stream = self.deepseek_client.chat.completions.create(
            model=model_config["model_id"],
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _chat_groq_stream(self, user_content: str, history: list, model_config: dict):
        """Stream chat using Groq API."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for h in history:
            if h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": user_content})

        stream = self.groq_client.chat.completions.create(
            model=model_config["model_id"],
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _chat_grok_stream(self, user_content: str, history: list, model_config: dict):
        """Stream chat using xAI Grok API (OpenAI-compatible)."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for h in history:
            if h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": user_content})

        stream = self.grok_client.chat.completions.create(
            model=model_config["model_id"],
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _chat_openrouter_stream(self, user_content: str, history: list, model_config: dict):
        """Stream chat using OpenRouter API (OpenAI-compatible)."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for h in history:
            if h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": user_content})

        stream = self.openrouter_client.chat.completions.create(
            model=model_config["model_id"],
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ===== .env Reload Function =====

    def reload_env_keys(self) -> str:
        """Reload API keys from .env file and reinitialize clients."""
        if not self.env_path.exists():
            return "❌ .env file not found"

        try:
            # Reload environment variables
            load_dotenv(self.env_path, override=True)

            # Reinitialize API clients
            openai_key = os.getenv("OPENAI_API_KEY", "").strip()
            self.openai_client = OpenAI() if openai_key else None

            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            self.anthropic_client = Anthropic() if anthropic_key else None

            deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            self.deepseek_client = OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com"
            ) if deepseek_key else None

            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            if GROQ_AVAILABLE and groq_key:
                self.groq_client = Groq()
            else:
                self.groq_client = None

            google_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if GOOGLE_AVAILABLE and google_key:
                self.google_client = genai.Client(api_key=google_key)
            else:
                self.google_client = None

            # xAI Grok client
            grok_key = os.getenv("GROK_API_KEY", "").strip()
            self.grok_client = OpenAI(
                api_key=grok_key,
                base_url="https://api.x.ai/v1"
            ) if grok_key else None

            # OpenRouter client
            openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            self.openrouter_client = OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            ) if openrouter_key else None

            # Build status message
            available = []
            if self.openai_client:
                available.append("OpenAI")
            if self.anthropic_client:
                available.append("Anthropic")
            if self.google_client:
                available.append("Google")
            if self.deepseek_client:
                available.append("DeepSeek")
            if self.groq_client:
                available.append("Groq")
            if self.grok_client:
                available.append("Grok")
            if self.openrouter_client:
                available.append("OpenRouter")

            if available:
                return f"✅ Reloaded: {', '.join(available)}"
            return "⚠️ No API keys configured"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    # Custom CSS for scrollable sources list
    CUSTOM_CSS = """
#sources-checkbox-container {
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 8px;
}
"""

    def create_interface(self) -> gr.Blocks:
        """Create the NotebookLM-style interface."""
        with gr.Blocks(title="My Research Notebook") as interface:
            gr.Markdown("# My Research Notebook")

            with gr.Row():
                # ===== Left Panel: Sources & Settings =====
                with gr.Column(scale=1):
                    # System Prompt Section (collapsed by default)
                    with gr.Accordion("📋 Project Instructions", open=False):
                        system_prompt_box = gr.Textbox(
                            label="System Prompt",
                            value=self.DEFAULT_SYSTEM_PROMPT,
                            lines=6,
                            max_lines=10,
                            placeholder="Enter project instructions and guidelines...",
                        )
                        with gr.Row():
                            update_prompt_btn = gr.Button("Update", size="sm", scale=1)
                            prompt_file = gr.File(
                                label=None,
                                file_types=[".md", ".txt"],
                                type="filepath",
                                scale=1,
                                min_width=120,
                            )
                        prompt_status = gr.Markdown(value="")

                    gr.Markdown("---")

                    # Sources Section with CheckboxGroup
                    gr.Markdown("### 📚 Sources")
                    stats_display = gr.Markdown(value=self.get_stats_text())

                    # Compact control row: Add + Select All + None + Delete
                    with gr.Row():
                        source_file = gr.UploadButton(
                            "➕",
                            file_types=[".pdf", ".md", ".txt", ".xml", ".docx"],
                            file_count="multiple",
                            size="sm",
                            scale=0,
                            min_width=50,
                        )
                        select_all_btn = gr.Button("✅ All", size="sm", scale=1)
                        deselect_all_btn = gr.Button("⬜ None", size="sm", scale=1)
                        delete_selected_btn = gr.Button("🗑️", variant="stop", size="sm", scale=0, min_width=40)

                    add_status = gr.Markdown(value="")

                    # Initialize CheckboxGroup with current choices at creation time
                    # (Gradio 6.x compatibility - interface.load may not trigger reliably)
                    initial_choices = self.get_sources_choices()
                    sources_checkbox = gr.CheckboxGroup(
                        choices=initial_choices,
                        label="Select sources for RAG context",
                        value=[],
                        interactive=True,
                        elem_id="sources-checkbox-container",
                    )

                    sources_status = gr.Markdown(value="")

                    gr.Markdown("---")

                    # .env Loader (simplified)
                    with gr.Accordion("⚙️ API Settings", open=False):
                        load_env_btn = gr.Button("🔄 Reload API Keys from .env", size="sm")
                        env_status = gr.Markdown("")

                # ===== Right Panel: Chat =====
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Chat")

                    # Model selection
                    with gr.Row():
                        model_selector = gr.Dropdown(
                            choices=[(v["name"], k) for k, v in MODELS.items()],
                            value="gpt-4o-mini",
                            label="Model",
                            scale=2,
                        )
                        use_sources_cb = gr.Checkbox(
                            label="Search KB",
                            value=True,
                            scale=0,
                        )

                    # Temporary context status
                    temp_status = gr.Markdown(value="📎 No temporary file loaded")

                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=600,
                    )

                    # Input area
                    with gr.Row():
                        temp_file = gr.UploadButton(
                            "📎",
                            file_types=[".pdf", ".md", ".txt", ".xml", ".docx"],
                            file_count="multiple",
                            size="sm",
                            scale=0,
                            min_width=50,
                        )
                        msg_input = gr.Textbox(
                            label="",
                            placeholder="",
                            scale=4,
                            show_label=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=0)

                    with gr.Row():
                        clear_context_btn = gr.Button("🗑️ Temp file", size="sm")
                        clear_chat_btn = gr.Button("🗑️ Chat", size="sm")

            # ===== Event Handlers =====

            # Update system prompt
            update_prompt_btn.click(
                fn=self.update_system_prompt,
                inputs=[system_prompt_box],
                outputs=[prompt_status],
            )

            # Load system prompt from file (triggered when file is selected)
            prompt_file.change(
                fn=self.load_system_prompt_file,
                inputs=[prompt_file],
                outputs=[system_prompt_box, prompt_status],
            )

            # Select all sources (refresh choices and select all)
            def select_all_sources():
                choices = self.get_sources_choices()
                all_ids = self.get_all_doc_ids()
                return gr.update(choices=choices, value=all_ids)

            select_all_btn.click(
                fn=select_all_sources,
                outputs=[sources_checkbox],
            )

            # Deselect all sources (refresh choices and clear selection)
            def deselect_all_sources():
                choices = self.get_sources_choices()
                return gr.update(choices=choices, value=[])

            deselect_all_btn.click(
                fn=deselect_all_sources,
                outputs=[sources_checkbox],
            )

            # Delete selected sources
            def delete_and_update_choices(selected_doc_ids):
                status, choices, stats = self.delete_selected_sources(selected_doc_ids)
                return status, gr.update(choices=choices, value=[]), stats

            delete_selected_btn.click(
                fn=delete_and_update_choices,
                inputs=[sources_checkbox],
                outputs=[sources_status, sources_checkbox, stats_display],
            )

            # Add source - update status, checkbox choices, and stats
            def add_source_and_refresh(file_paths, progress=gr.Progress()):
                status, _ = self.add_source(file_paths, progress)
                choices = self.get_sources_choices()
                stats = self.get_stats_text()
                return status, gr.update(choices=choices), stats

            source_file.upload(
                fn=add_source_and_refresh,
                inputs=[source_file],
                outputs=[add_status, sources_checkbox, stats_display],
            )

            # Upload temp file
            temp_file.upload(
                fn=self.upload_temp_file,
                inputs=[temp_file],
                outputs=[temp_status],
            )

            # Clear temp context
            clear_context_btn.click(
                fn=self.clear_temp_context,
                outputs=[temp_status],
            )

            # Clear chat
            clear_chat_btn.click(
                fn=lambda: [],
                outputs=[chatbot],
            )

            # Send message with selected sources (streaming)
            send_btn.click(
                fn=self.chat_stream,
                inputs=[msg_input, chatbot, model_selector, use_sources_cb, sources_checkbox],
                outputs=[chatbot, msg_input],
            )

            # Enter key to send (streaming)
            msg_input.submit(
                fn=self.chat_stream,
                inputs=[msg_input, chatbot, model_selector, use_sources_cb, sources_checkbox],
                outputs=[chatbot, msg_input],
            )

            # .env reload handler
            load_env_btn.click(
                fn=self.reload_env_keys,
                outputs=[env_status],
            )

            # Refresh sources on page load (fallback for dynamic updates)
            # Note: Choices are also initialized at creation time for Gradio 6.x compatibility
            def refresh_on_load():
                choices = self.get_sources_choices()
                stats = self.get_stats_text()
                return gr.update(choices=choices, value=[]), stats

            interface.load(
                fn=refresh_on_load,
                outputs=[sources_checkbox, stats_display],
            )

        return interface

    def launch(self, **kwargs):
        """Launch the interface."""
        interface = self.create_interface()
        # Pass CSS to launch() for Gradio 6.x compatibility
        interface.launch(css=self.CUSTOM_CSS, **kwargs)


def main():
    """Entry point."""
    ui = NotebookUI()
    ui.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
