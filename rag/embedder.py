"""
HuggingFace embedding functions for the DevMind RAG pipeline.

All embedding logic is encapsulated here so the rest of the pipeline
never has to touch the HuggingFace client directly.

Functions
---------
create_hf_client                    -- build an InferenceClient from a token
normalize_embedding_output          -- normalise raw HF output → 2-D float32 array
hf_feature_extraction_with_retries  -- call HF with exponential-backoff retries
embed_texts                         -- batch-embed a list of strings
embed_query                         -- embed a single query string
"""

import os
import time

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_EMBEDDING_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
BATCH_SIZE = 8


def _log(msg: str) -> None:
    print(f"[embedder] {msg}", flush=True)


# ------------------------------------------------------------------
# CLIENT FACTORY
# ------------------------------------------------------------------

def create_hf_client(token: str = HF_TOKEN) -> InferenceClient:
    """Return a configured HuggingFace InferenceClient."""
    return InferenceClient(provider="hf-inference", api_key=token)


# ------------------------------------------------------------------
# NORMALISATION
# ------------------------------------------------------------------

def normalize_embedding_output(
    raw_output,
    expected_count: int,
) -> np.ndarray:
    """
    Convert the raw HuggingFace feature-extraction response into a clean
    2-D float32 numpy array of shape (expected_count, embedding_dim).

    The HF API can return embeddings in several shapes depending on the
    model and input size. This function handles all known cases:

    - 1-D  [embedding_dim]                     → single embedding
    - 2-D  [expected_count, embedding_dim]      → standard batch output
    - 2-D  [tokens, embedding_dim]              → token-level output for 1 input
    - 3-D  [batch, tokens, embedding_dim]       → token-level output for batch
    """
    arr = np.array(raw_output, dtype="float32")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[0] == expected_count:
            pass  # already correct shape
        elif expected_count == 1:
            # token-level output for a single input — mean-pool across tokens
            arr = arr.mean(axis=0, keepdims=True)
        else:
            raise ValueError(
                f"Unexpected 2-D embedding shape: {arr.shape}, "
                f"expected_count={expected_count}"
            )
    elif arr.ndim == 3:
        # token-level output for a batch — mean-pool across token dimension
        arr = arr.mean(axis=1)
    else:
        raise ValueError(f"Unexpected embedding dimensions: {arr.ndim}")

    if arr.shape[0] != expected_count:
        raise ValueError(
            f"Embedding count mismatch. "
            f"Expected {expected_count}, got {arr.shape[0]}"
        )

    return arr.astype("float32")


# ------------------------------------------------------------------
# EMBEDDING WITH RETRIES
# ------------------------------------------------------------------

def hf_feature_extraction_with_retries(
    hf_client: InferenceClient,
    inputs,
    expected_count: int,
    max_retries: int = 5,
) -> np.ndarray:
    """
    Call the HuggingFace cloud embedding model with exponential-backoff retries.

    Parameters
    ----------
    hf_client      : pre-built InferenceClient
    inputs         : str or list[str] — text(s) to embed
    expected_count : number of embeddings expected in the response
    max_retries    : maximum number of attempts before re-raising the error

    Returns
    -------
    Float32 numpy array of shape (expected_count, embedding_dim)
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = hf_client.feature_extraction(inputs, model=HF_EMBEDDING_MODEL)
            return normalize_embedding_output(
                raw_output=result, expected_count=expected_count
            )
        except Exception as exc:
            last_error = exc
            _log(f"Embedding call failed (attempt {attempt}/{max_retries}): {exc}")
            if attempt == max_retries:
                raise
            wait = attempt * 3
            _log(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"Embedding failed after {max_retries} attempts: {last_error}"
    )


# ------------------------------------------------------------------
# BATCH TEXT EMBEDDING
# ------------------------------------------------------------------

def embed_texts(
    hf_client: InferenceClient,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    progress_callback=None,
) -> np.ndarray:
    """
    Embed a list of text strings in batches.

    Parameters
    ----------
    hf_client         : pre-built InferenceClient
    texts             : list of strings to embed
    batch_size        : number of texts per API call
    progress_callback : optional callable(current_batch, total_batches)
                        invoked after each batch completes

    Returns
    -------
    Float32 numpy array of shape (len(texts), embedding_dim)
    """
    all_embeddings: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_num = start // batch_size + 1
        _log(f"  batch {batch_num}/{total_batches} ({len(batch)} items)...")

        embeddings = hf_feature_extraction_with_retries(
            hf_client=hf_client,
            inputs=batch,
            expected_count=len(batch),
        )
        all_embeddings.append(embeddings)

        if progress_callback is not None:
            progress_callback(batch_num, total_batches)

    return np.vstack(all_embeddings).astype("float32")


# ------------------------------------------------------------------
# SINGLE QUERY EMBEDDING
# ------------------------------------------------------------------

def embed_query(hf_client: InferenceClient, query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns
    -------
    Float32 numpy array of shape (1, embedding_dim)
    """
    embedding = hf_feature_extraction_with_retries(
        hf_client=hf_client,
        inputs=query,
        expected_count=1,
    )
    return embedding.astype("float32")
