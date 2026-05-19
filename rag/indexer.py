"""
FAISS vector index creation and similarity retrieval for DevMind.

Functions
---------
create_faiss_index -- build an L2-normalised inner-product index from embeddings
retrieve           -- embed a query and return the top-k matching chunks
"""

import faiss
import numpy as np
from huggingface_hub import InferenceClient

from rag.embedder import embed_query

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

TOP_K = 3
MIN_SCORE = 0.3


# ------------------------------------------------------------------
# INDEX CREATION
# ------------------------------------------------------------------

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (inner-product) index from pre-computed embeddings.

    Vectors are L2-normalised before indexing so that inner-product search
    is equivalent to cosine similarity search.

    Parameters
    ----------
    embeddings : float32 numpy array of shape (n_docs, embedding_dim)

    Returns
    -------
    A populated faiss.IndexFlatIP ready for .search() calls
    """
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


# ------------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------------

def retrieve(
    index: faiss.IndexFlatIP,
    chunks: list[str],
    sources: list[str],
    hf_client: InferenceClient,
    query: str,
    k: int = TOP_K,
    min_score: float = MIN_SCORE,
) -> list[dict]:
    """
    Embed *query*, search *index*, and return the top-k matching chunks
    whose cosine-similarity score meets the *min_score* threshold.

    The query vector is L2-normalised to match the normalisation applied
    during index construction, making scores interpretable as cosine
    similarity values in [−1, 1].

    Parameters
    ----------
    index     : populated FAISS index returned by create_faiss_index
    chunks    : list of text strings (same order as when the index was built)
    sources   : list of source filenames parallel to *chunks*
    hf_client : HuggingFace InferenceClient used for query embedding
    query     : the user's natural-language question
    k         : number of results to return
    min_score : minimum cosine-similarity score; results below this are discarded

    Returns
    -------
    List of dicts, each containing:
        text   -- the retrieved chunk string
        source -- filename the chunk came from (or None if unavailable)
        score  -- cosine-similarity score as a Python float
    """
    query_embedding = embed_query(hf_client, query)
    faiss.normalize_L2(query_embedding)

    scores, indexes = index.search(query_embedding, k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indexes[0]):
        if idx == -1:
            continue
        if float(score) < min_score:
            continue
        results.append(
            {
                "text": chunks[idx],
                "source": sources[idx] if idx < len(sources) else None,
                "score": float(score),
            }
        )
    return results
