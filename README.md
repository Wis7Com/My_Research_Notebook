# My Research Notebook - Local RAG Application

![My Research Notebook](docs/my_research_notebook_thumbnail_v3.png)

A general-purpose RAG (Retrieval-Augmented Generation) application for research document drafting. Upload PDFs and other documents, build a searchable vector database, and chat with multiple LLM providers using your knowledge base as context.

## Features

### Core Capabilities
- **Multi-Format Support**: PDF, Word (.docx), Markdown (.md), Plain Text (.txt), XML
- **Document Processing**: PyMuPDF for PDFs, python-docx for Word documents
- **Vector Database**: ChromaDB with OpenAI embeddings for semantic search
- **Citation Management**: Automatic BibTeX generation with page-level tracking
- **MCP Server**: Claude Code CLI integration for direct querying
- **Multi-Provider Support**: OpenAI, Anthropic, Google, DeepSeek, Groq, xAI, OpenRouter

### Gradio UI Panels
- **Sources Panel**: Upload documents (PDF, DOCX, TXT, MD, XML), select for RAG context, view chunk statistics
- **Chat Panel**: Multi-provider LLM chat with streaming responses, model selection, and conversation history
- **Project Instructions**: Customizable system prompt editor with file upload support for .md/.txt files

## Quick Start

### 1. Setup Environment (uv)

```bash
uv venv .venv
uv pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
DEEPSEEK_API_KEY=your-deepseek-key
GROQ_API_KEY=your-groq-key
GROK_API_KEY=your-grok-key
OPENROUTER_API_KEY=your-openrouter-key
```

Only `OPENAI_API_KEY` is required for embeddings. Other keys are optional based on which LLM providers you want to use.

### 3. Add Source Documents

Create a `sources_raw/` directory and add your documents (supports PDF, DOCX, TXT, MD, XML):
```bash
mkdir -p sources_raw
```

You can organize documents in subdirectories as needed (e.g., by topic, date, or type). The application will recursively discover all documents:
```
sources_raw/
├── your_topic_1/
├── your_topic_2/
└── ...
```

The `--phase` filter accepts partial folder name matches for targeted searches.

### 4. Build the Vector Database

Process all documents:
```bash
uv run python -m app.main build
```

Options:
- `--clear`: Clear existing data before building
- `--fallback`: Use PyMuPDF parser instead of Docling

### 5. Use the Application

**Launch Web UI:**
```bash
uv run python -m app.main ui
```
Open http://127.0.0.1:7860 in your browser.

**Search from Command Line:**
```bash
uv run python -m app.main search "international law principles"
uv run python -m app.main search "regulatory framework" --phase "meeting_1" -n 10
```

**View Statistics:**
```bash
uv run python -m app.main stats
```

**Add Single Document:**
```bash
uv run python -m app.main add path/to/document.pdf
uv run python -m app.main add path/to/notes.docx
```

**Run MCP Server Directly:**
```bash
uv run python -m app.main mcp
```

## Claude Code MCP Integration

### Configure MCP Server

Add to your Claude Code settings (`~/.claude.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "research-rag": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server.server"],
      "cwd": "/path/to/your/research-notebook"
    }
  }
}
```

### Available MCP Tools

Once configured, Claude Code can use these tools:

- `search_documents`: Semantic search with metadata filtering
- `get_citation`: Get BibTeX citation for a document
- `verify_citation`: Verify statements against source documents
- `list_documents`: List all documents in the knowledge base
- `get_document_context`: Get full content of a specific document
- `get_database_stats`: Get knowledge base statistics

### Example Usage in Claude Code

```
Use the search_documents tool to find information about
regulatory frameworks in the knowledge base.
```

## Architecture

```
Source Documents (sources_raw/)
    │  Supports: PDF, DOCX, TXT, MD, XML
    ▼
┌─────────────────────────────────────┐
│  Document Parser (app/parsers/)     │
│  - PDF: PyMuPDF text extraction     │
│  - DOCX: python-docx                │
│  - TXT/MD/XML: Native parsing       │
│  - Character-based chunking         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Vector DB (app/vectordb/)          │
│  - ChromaDB + OpenAI embeddings     │
│  - Metadata: page, section, folder  │
│  - Persistent at data/chroma_db/    │
└─────────────────────────────────────┘
    │
    ├──► MCP Server (mcp_server/)
    │    └─► Claude Code CLI tools
    │
    ├──► Gradio UI (app/ui/)
    │    └─► Document upload/search/chat
    │
    └──► Citations (app/citations/)
         └─► BibTeX generation
```

## Project Structure

```
├── app/                    # Main application
│   ├── parsers/            # Document processing (PDF, DOCX, TXT, MD, XML)
│   ├── vectordb/           # ChromaDB + embeddings
│   ├── citations/          # BibTeX management
│   └── ui/                 # Gradio interface
├── mcp_server/             # MCP server for Claude Code
├── scripts/                # Utility scripts
├── data/                   # Generated data
│   ├── chroma_db/          # Vector database
│   └── citations/          # BibTeX files
├── sources_raw/            # Your source documents (create this directory)
└── outputs/                # Generated outputs
```

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | GPT-5.x, GPT-4o, GPT-4o Mini, o3/o1 reasoning |
| Anthropic | Claude Sonnet 4, Opus 4, 3.7, 3.5 |
| Google | Gemini 2.0/1.5 |
| DeepSeek | R1, V3 |
| Groq | Llama 3.3, Mixtral, Gemma2 (free tier) |
| xAI | Grok 2, Grok 2 Vision (free tier) |
| OpenRouter | Llama 3.3 70B, Gemma 2 9B, Mistral 7B (free tier) |

## Chunking Strategy

- Chunk size: ~1500 tokens with 200 token overlap
- Sections preserved from markdown headings
- Footnotes attached to parent paragraphs
- Page numbers estimated from document position

## Citation Format

Citations are generated in BibTeX format:
```bibtex
@techreport{example2024report,
  title = {Research Report Title},
  author = {Author Name},
  year = {2024},
  note = {Document ID: abc123def456},
}
```

## Troubleshooting

**Docling installation issues:**
Use the fallback parser:
```bash
uv run python -m app.main build --fallback
```

**OpenAI API errors:**
Verify your API key in `.env` and check rate limits.

**Empty search results:**
Ensure the database is built:
```bash
uv run python -m app.main stats
```

## License

MIT License
