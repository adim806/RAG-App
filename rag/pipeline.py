"""
Full RAG pipeline for DevMind.

The RAGEngine class is the single public surface used by the Flask web app.
It coordinates loader → embedder → indexer → Gemini and holds all state in
memory so that only one initialisation pass is needed per server process.

Classes
-------
RAGEngine -- initialise once at startup; call .answer() per user question
"""

import os
import threading

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.embedder import BATCH_SIZE, create_hf_client, embed_texts
from rag.indexer import TOP_K, create_faiss_index, retrieve
from rag.loader import DATA_FOLDER, load_documents, setup_nltk

load_dotenv()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3-flash-preview"


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


# ------------------------------------------------------------------
# RAG ENGINE
# ------------------------------------------------------------------

class RAGEngine:
    """
    Encapsulates the entire RAG pipeline as a single in-memory object.

    Lifecycle
    ---------
    1. Instantiate once (e.g. at Flask app startup).
    2. Call .initialise() — this is blocking and may take 1–2 minutes
       the first time because it embeds all documents.  It is idempotent:
       subsequent calls return immediately.
    3. Call .answer(question, history) for each user turn.

    Thread safety
    -------------
    .initialise() is protected by a lock so it is safe to call from a
    background thread or from multiple concurrent requests.  All other
    methods are read-only after initialisation and therefore thread-safe.
    """

    def __init__(self, data_folder: str = DATA_FOLDER) -> None:
        self.data_folder = data_folder
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.index = None
        self.ready: bool = False
        self.status: str = "not_initialised"
        self.progress: dict = {"current": 0, "total": 0}

        self._lock = threading.Lock()

        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        self.hf_client = create_hf_client()

    # ------------------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Load documents, build embeddings, and build the FAISS index.

        This method is idempotent — calling it a second time is a no-op.
        """
        with self._lock:
            if self.ready:
                return

            self.status = "downloading_nltk"
            _log("Setting up NLTK tokenizer...")
            setup_nltk()

            self.status = "loading_documents"
            _log(f"Loading documents from '{self.data_folder}'...")
            self.chunks, self.sources = load_documents(self.data_folder)
            _log(
                f"Loaded {len(self.chunks)} sentences "
                f"from {len(set(self.sources))} file(s)."
            )

            self.status = "embedding_documents"
            _log("Embedding documents via HuggingFace...")

            def _progress(current: int, total: int) -> None:
                self.progress = {"current": current, "total": total}

            embeddings = embed_texts(
                hf_client=self.hf_client,
                texts=self.chunks,
                batch_size=BATCH_SIZE,
                progress_callback=_progress,
            )

            self.status = "building_index"
            _log("Building FAISS index...")
            self.index = create_faiss_index(embeddings)

            self.ready = True
            self.status = "ready"
            _log(f"Ready — {self.index.ntotal} vectors indexed.")

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """Return the top-k chunks most relevant to *query*.

        Each result is a dict with keys: text, source, score.

        Raises RuntimeError if the engine has not been initialised.
        """
        if not self.ready:
            raise RuntimeError("RAG engine is not ready yet.")

        return retrieve(
            index=self.index,
            chunks=self.chunks,
            sources=self.sources,
            hf_client=self.hf_client,
            query=query,
            k=k,
        )

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------

    def ask_gemini(
        self,
        context: str,
        question: str,
        history: list[dict] | None = None,
    ) -> str:
        """
        Call Gemini with retrieved context, conversation history, and the
        user's latest question.

        Parameters
        ----------
        context  : newline-joined retrieved chunks
        question : the user's current question
        history  : list of {"role": "user"|"assistant", "content": str}
                   entries representing the conversation so far (excluding
                   the current question)

        Returns
        -------
        The model's answer as a stripped string.
        """
        history_text = ""
        if history:
            lines = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            history_text = "\n".join(lines)

        prompt = f"""You are a helpful RAG assistant for a software engineering team.

Use the provided context and the previous conversation to answer the user's
latest question.

Rules:
1. Answer using ONLY the provided context. Do NOT use general knowledge.
2. If the context does not contain enough information, reply EXACTLY:
   "I don't have enough information in the internal documents to answer this question."
3. Keep the answer simple and clear.
4. Do not invent or assume facts that are not explicitly stated in the context.
5. Use the conversation history only to resolve references like "it" or
   "the previous one" — never invent earlier turns.

Conversation so far:
{history_text if history_text else "(no previous messages)"}

Context retrieved from documents:
{context}

User's latest question:
{question}

Answer:
"""

        response = self.gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()

    # ------------------------------------------------------------------
    # HIGH-LEVEL HELPER
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = TOP_K,
    ) -> dict:
        """
        Run the full retrieve → generate flow for a single user turn.

        Parameters
        ----------
        question : the user's current question
        history  : conversation history (see ask_gemini for format)
        k        : number of chunks to retrieve

        Returns
        -------
        dict with keys:
            answer  -- the model's answer string
            context -- list of retrieved chunk dicts (text, source, score)

        Raises RuntimeError if the engine has not been initialised.
        """
        if not self.ready:
            raise RuntimeError("RAG engine is not ready yet.")

        retrieved = self.retrieve(question, k=k)
        context = "\n".join(item["text"] for item in retrieved)
        answer_text = self.ask_gemini(
            context=context,
            question=question,
            history=history or [],
        )
        return {
            "answer": answer_text,
            "context": retrieved,
        }
