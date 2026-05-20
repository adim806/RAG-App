# DevMind — RAG-Powered Internal Knowledge Base

A Retrieval-Augmented Generation (RAG) web application that serves as an
intelligent assistant for a software engineering team's internal documentation.
Engineers can ask natural-language questions and receive grounded answers sourced
exclusively from the company's knowledge base.

## Topic & Motivation

The chosen domain is **internal engineering documentation** — the kind of
knowledge base that every software team accumulates: coding standards, Git
workflow rules, API guidelines, testing policies, system architecture, and
onboarding guides.

This topic was selected because:

- It mirrors a real professional need — new engineers constantly look up team
  conventions and best practices.
- The documents are well-structured and contain factual, verifiable content,
  which makes it straightforward to validate retrieval accuracy.
- It demonstrates a practical RAG use case (company knowledge search) that
  translates directly to production environments.

## Architecture Overview

```
User ──▶ Flask Web UI ──▶ POST /api/ask
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             FAISS Retrieval       Conversation
             (top-k chunks)         History (SQLite)
                    │                    │
                    └────────┬───────────┘
                             ▼
                     Gemini LLM Prompt
                     (context + history + question)
                             │
                             ▼
                      Grounded Answer
```

| Component | Technology |
|-----------|-----------|
| Web framework | Flask 3 |
| Embedding model | `ibm-granite/granite-embedding-97m-multilingual-r2` (HuggingFace Inference API) |
| Vector store | FAISS (`IndexFlatIP`, cosine similarity) |
| LLM | Google Gemini (`gemini-3-flash-preview`) |
| Conversation store | SQLite |
| Sentence tokenizer | NLTK (`sent_tokenize`) |
| PDF text extraction | PyMuPDF (`fitz`) |
| Containerisation | Docker |

## Data Source

The knowledge base consists of **7 documents** (6 text files + 1 PDF) stored in
the `data/` directory.  They simulate a realistic set of internal engineering
docs for a fictional company called "DevMind":

| File | Content |
|------|---------|
| `architecture.txt` | Microservices layout, AWS/K8s deployment, data stores, CI/CD, observability |
| `api_guidelines.txt` | REST conventions, HTTP status codes, authentication, pagination, validation |
| `coding_standards.txt` | PEP 8, naming conventions, type hints, import ordering, formatting rules |
| `git_workflow.txt` | Branch naming, Conventional Commits, PR requirements, merge strategies |
| `onboarding.txt` | Prerequisites, environment setup, first-week checklist, team practices |
| `testing_policy.txt` | Coverage requirements, test types, naming conventions, flaky-test policy |
| `devmind_security_guidelines.pdf` | Password hashing, JWT storage, input validation, encryption, audit logging |

These documents were written to be factually consistent and internally
cross-referenced, providing a realistic corpus for semantic retrieval.  The PDF
file demonstrates that the system supports multiple document formats.

## Chunking Strategy

Documents are split into **sentence-level chunks** using NLTK's
`sent_tokenize`.  Each sentence becomes one vector in the FAISS index, paired
with its source filename.

**Why sentence-level chunking?**

- Internal docs are written in standalone declarative sentences (e.g. "Use
  snake_case for Python variables"), each carrying one distinct fact.
- Fine-grained chunks reduce noise — the LLM receives only the most relevant
  sentences rather than large paragraphs that may contain unrelated information.
- Simple and deterministic — the same data always produces the same index.

**Trade-offs acknowledged:**

- Very short sentences may lack context on their own.
- An alternative section-based chunker (`loader.chunk_by_section`) is included
  in the codebase but not used by default; it groups text by section headers for
  coarser retrieval when broader context is preferred.

## Embedding Model

The project uses **IBM Granite Embedding 97M Multilingual R2**
(`ibm-granite/granite-embedding-97m-multilingual-r2`) via the HuggingFace
Inference API.

**Why this model?**

- Lightweight (97M parameters) — fast inference via the free HF API tier.
- Multilingual support — handles English technical text well and can be extended
  to other languages.
- Produces high-quality dense vectors for semantic similarity tasks.

Embeddings are generated in **batches of 8** with exponential-backoff retries
(up to 5 attempts) to handle transient API errors gracefully.

## FAISS Indexing

- **Index type:** `IndexFlatIP` (flat inner-product index).
- All document vectors are **L2-normalised** before indexing, which makes
  inner-product search equivalent to **cosine similarity** search.
- At query time the query vector is also L2-normalised before searching.
- **Top-K = 3** results are returned per query.
- A **minimum score threshold of 0.3** filters out low-relevance results,
  ensuring the LLM only receives genuinely related context.
- The index is rebuilt in memory on every server start (~1-2 min on first run).

## RAG Pipeline

1. **User submits a question** via the web UI (`POST /api/ask`).
2. **Retrieval** — the question is embedded and searched against the FAISS index;
   the top-3 chunks above the relevance threshold are returned.
3. **Prompt construction** — retrieved chunks are joined as context and combined
   with conversation history and the user's question into a structured prompt.
4. **LLM generation** — Gemini generates a grounded answer with `temperature=0.3`.
5. **Persistence** — both the question and the answer (with source metadata) are
   saved to SQLite for session continuity.

### Hallucination Reduction

- The system prompt instructs the model to answer **only** from the provided
  context and to explicitly refuse when information is insufficient.
- `temperature=0.3` minimises creative generation.
- The relevance threshold prevents low-quality context from reaching the LLM.
- `max_output_tokens=500` keeps answers concise and focused.

### Edge Case Handling

| Scenario | Behaviour |
|----------|-----------|
| Empty question | Returns 400 error |
| Engine not ready | Returns 503 with loading status |
| Session not found | Returns 404 error |
| Pipeline exception | Returns 500 with error message |
| No relevant results (all below threshold) | Empty context → LLM states lack of information |
| Irrelevant question (out-of-domain) | LLM responds with "I don't have enough information" |

## Web Application

- **Dark-themed UI** with a professional design (Inter + JetBrains Mono fonts).
- **Sidebar** with session management (create, switch, delete conversations).
- **Welcome screen** with suggestion chips for common questions.
- **Chat interface** with user/assistant message bubbles.
- **Source display** — each answer shows expandable source documents with
  similarity scores.
- **Loading indicators** — full-screen init overlay with progress bar during
  startup; animated thinking dots while waiting for responses.
- **Error handling** — network and API errors shown inline in the chat.
- **Responsive design** — sidebar collapses on mobile screens.

## Installation & Running

### Option A — Docker (recommended)

The easiest way to run the application.  Only **Docker** and an internet
connection are required.

```bash
cd RAG-App

# 1. Build the image
docker build -t devmind-rag .

# 2. Run the container (pass API keys via .env)
docker run -d --name devmind -p 5000:5000 --env-file .env devmind-rag
```

Open **http://localhost:5000** in your browser.  The knowledge base embedding
takes ~1-2 minutes on first launch; a loading overlay shows progress.

Useful Docker commands:

```bash
docker logs -f devmind       # watch logs in real time
docker stop devmind           # stop the container
docker start devmind          # restart the container
docker rm -f devmind          # remove and re-create if needed
```

### Option B — Local Python

#### Prerequisites

- Python 3.11+
- Internet connection (for HuggingFace and Gemini API calls)

#### Setup

```bash
cd RAG-App

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys — create a .env file with:
#   HF_TOKEN=your_huggingface_token
#   GEMINI_API_KEY=your_gemini_api_key
```

#### Running the Application

```bash
python run.py
```

The app starts at **http://localhost:5000**.  On first launch, the knowledge base
embedding takes ~1-2 minutes; a loading overlay shows progress.

### Running Tests

```bash
pytest tests/test_queries.py -v -s
```

Tests require live API access (HuggingFace + Gemini) and take a few minutes on
first run due to embedding generation.

## Test Queries & Validation

The test suite (`tests/test_queries.py`) validates the full RAG pipeline with
**8 documented test cases**:

| # | Question | Expected Behaviour | Source Document |
|---|----------|--------------------|-----------------|
| 1 | What naming convention should I use for Python functions? | Mentions `snake_case` or PEP 8 | `coding_standards.txt` |
| 2 | How should I name a feature branch in Git? | Mentions `feature/` prefix | `git_workflow.txt` |
| 3 | What HTTP status code should I return for a validation error? | Mentions 400 or 422 | `api_guidelines.txt` |
| 4 | What is the minimum required unit test coverage percentage? | Mentions 80% | `testing_policy.txt` |
| 5 | How do I set up my local development environment? | Mentions Docker, pip, or clone | `onboarding.txt` |
| 6 | What database technology does the DevMind platform use? | Mentions PostgreSQL or Redis | `architecture.txt` |
| 7 | What about hotfix branches? (with conversation history) | Coherent multi-turn answer | `git_workflow.txt` |
| 8 | What is the capital of France? (irrelevant) | Refuses or states lack of information | N/A |

## Project Structure

```
RAG-App/
├── run.py                  # Entry point — starts Flask on port 5000
├── database.py             # SQLite session & message persistence
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image build recipe
├── .dockerignore           # Files excluded from Docker image
├── .env                    # API keys (not committed)
├── .gitignore
├── README.md
├── data/
│   ├── devmind_security_guidelines.pdf
│   ├── api_guidelines.txt
│   ├── architecture.txt
│   ├── coding_standards.txt
│   ├── git_workflow.txt
│   ├── onboarding.txt
│   └── testing_policy.txt
├── rag/
│   ├── __init__.py
│   ├── loader.py           # Document loading & sentence chunking
│   ├── embedder.py         # HuggingFace embedding (batch, retries)
│   ├── indexer.py          # FAISS index creation & retrieval
│   └── pipeline.py         # RAGEngine — orchestrates the full pipeline
├── web/
│   ├── app.py              # Flask routes & API endpoints
│   ├── templates/
│   │   └── index.html      # Chat UI (HTML + inline JS)
│   └── static/
│       └── style.css       # Dark-theme stylesheet
└── tests/
    └── test_queries.py     # Integration tests (8 documented queries)
```
