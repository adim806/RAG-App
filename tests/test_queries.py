"""
DevMind RAG system — documented test queries.

Each test sends a real question through the full RAGEngine pipeline and
validates that the answer is non-empty and the retrieved context contains
at least one result.  Expected answer keywords are checked to give
confidence that the retrieval is pointing at the right document.

Run with:
    cd devmind
    pytest tests/test_queries.py -v

NOTE: These tests hit the HuggingFace cloud API and the Gemini API, so they
require a live internet connection and valid API keys.  They may take a few
minutes on the first run because the engine has to embed all documents.
"""

import sys
import os

# Ensure the devmind root is on sys.path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from rag.pipeline import RAGEngine  # noqa: E402


# ------------------------------------------------------------------
# SHARED FIXTURE — initialise the engine once for the test session
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine() -> RAGEngine:
    """
    Initialise a single RAGEngine for all tests.

    Embedding all documents is expensive (~1-2 min on first run) so we
    share one instance across the session.
    """
    e = RAGEngine()
    e.initialise()
    assert e.ready, "RAGEngine failed to initialise"
    return e


# ==================================================================
# TEST 1 — Python coding style
# ==================================================================

def test_python_naming_convention(engine: RAGEngine):
    """
    Question : What naming convention should I use for Python functions?
    Expected : Answer mentions snake_case from coding_standards.txt.
    """
    question = "What naming convention should I use for Python functions?"
    result   = engine.answer(question)

    answer  = result["answer"].lower()
    context = result["context"]

    assert answer, "Answer must not be empty"
    assert len(context) > 0, "At least one context chunk must be retrieved"

    # The answer should reference snake_case or PEP8
    assert "snake_case" in answer or "pep" in answer or "naming" in answer, (
        f"Expected naming convention keywords in answer.\nGot: {result['answer']}"
    )

    print("\n[TEST 1] Python naming convention")
    print(f"  Q: {question}")
    print(f"  A: {result['answer'][:300]}")
    print(f"  Sources: {[c['source'] for c in context]}")


# ==================================================================
# TEST 2 — Git branching
# ==================================================================

def test_git_feature_branch_naming(engine: RAGEngine):
    """
    Question : How should I name a feature branch?
    Expected : Answer mentions feature/ prefix from git_workflow.txt.
    """
    question = "How should I name a feature branch in Git?"
    result   = engine.answer(question)

    answer  = result["answer"].lower()
    context = result["context"]

    assert answer, "Answer must not be empty"
    assert len(context) > 0, "At least one context chunk must be retrieved"

    assert "feature/" in answer or "branch" in answer or "naming" in answer, (
        f"Expected branch naming info in answer.\nGot: {result['answer']}"
    )

    print("\n[TEST 2] Git feature branch naming")
    print(f"  Q: {question}")
    print(f"  A: {result['answer'][:300]}")
    print(f"  Sources: {[c['source'] for c in context]}")


# ==================================================================
# TEST 3 — API error codes
# ==================================================================

def test_api_validation_error_status_code(engine: RAGEngine):
    """
    Question : What HTTP status code should I return for a validation error?
    Expected : Answer mentions 400 or 422 from api_guidelines.txt.
    """
    question = "What HTTP status code should I return for a validation error?"
    result   = engine.answer(question)

    answer  = result["answer"]
    context = result["context"]

    assert answer, "Answer must not be empty"
    assert len(context) > 0, "At least one context chunk must be retrieved"

    assert "400" in answer or "422" in answer or "Bad Request" in answer, (
        f"Expected 400/422 status code in answer.\nGot: {answer}"
    )

    print("\n[TEST 3] API validation error status code")
    print(f"  Q: {question}")
    print(f"  A: {answer[:300]}")
    print(f"  Sources: {[c['source'] for c in context]}")


# ==================================================================
# TEST 4 — Test coverage requirement
# ==================================================================

def test_minimum_test_coverage(engine: RAGEngine):
    """
    Question : What is the minimum required unit test coverage?
    Expected : Answer mentions 80% from testing_policy.txt.
    """
    question = "What is the minimum required unit test coverage percentage?"
    result   = engine.answer(question)

    answer  = result["answer"]
    context = result["context"]

    assert answer, "Answer must not be empty"
    assert len(context) > 0, "At least one context chunk must be retrieved"

    assert "80" in answer or "coverage" in answer.lower(), (
        f"Expected 80% coverage info in answer.\nGot: {answer}"
    )

    print("\n[TEST 4] Minimum test coverage")
    print(f"  Q: {question}")
    print(f"  A: {answer[:300]}")
    print(f"  Sources: {[c['source'] for c in context]}")


# ==================================================================
# TEST 5 — Onboarding setup
# ==================================================================

def test_local_dev_setup(engine: RAGEngine):
    """
    Question : How do I set up my local development environment?
    Expected : Answer mentions Docker, pip, or prerequisites from onboarding.txt.
    """
    question = "How do I set up my local development environment as a new engineer?"
    result   = engine.answer(question)

    answer  = result["answer"].lower()
    context = result["context"]

    assert answer, "Answer must not be empty"
    assert len(context) > 0, "At least one context chunk must be retrieved"

    assert any(kw in answer for kw in ["docker", "pip", "python", "install", "clone"]), (
        f"Expected setup keywords in answer.\nGot: {result['answer']}"
    )

    print("\n[TEST 5] Local dev setup")
    print(f"  Q: {question}")
    print(f"  A: {result['answer'][:300]}")
    print(f"  Sources: {[c['source'] for c in context]}")


# ==================================================================
# TEST 6 — System architecture
# ==================================================================

def test_platform_database(engine: RAGEngine):
    """
    Question : What database does the DevMind platform use?
    Expected : Answer mentions PostgreSQL or Redis from architecture.txt.
    """
    question = "What database technology does the DevMind platform use?"
    result   = engine.answer(question)

    answer  = result["answer"].lower()
    context = result["context"]

    assert answer, "Answer must not be empty"
    assert len(context) > 0, "At least one context chunk must be retrieved"

    assert "postgresql" in answer or "postgres" in answer or "redis" in answer, (
        f"Expected database name in answer.\nGot: {result['answer']}"
    )

    print("\n[TEST 6] Platform database")
    print(f"  Q: {question}")
    print(f"  A: {result['answer'][:300]}")
    print(f"  Sources: {[c['source'] for c in context]}")


# ==================================================================
# TEST 7 — Conversation history (multi-turn)
# ==================================================================

def test_multi_turn_history(engine: RAGEngine):
    """
    Tests that passing history doesn't break the pipeline and the answer
    is still coherent.
    """
    history = [
        {"role": "user",      "content": "What is the branch naming convention?"},
        {"role": "assistant", "content": "Feature branches use the feature/ prefix."},
    ]
    question = "What about hotfix branches?"
    result   = engine.answer(question, history=history)

    answer  = result["answer"].lower()
    assert answer, "Answer must not be empty with history"
    assert len(result["context"]) > 0, "Context must be returned with history"

    print("\n[TEST 7] Multi-turn conversation")
    print(f"  Q: {question}")
    print(f"  A: {result['answer'][:300]}")
    print(f"  Sources: {[c['source'] for c in result['context']]}")


# ==================================================================
# TEST 8 — Irrelevant question (out-of-domain)
# ==================================================================

def test_irrelevant_question(engine: RAGEngine):
    """
    Question : What is the capital of France?
    Expected : The system should indicate it does not have enough
               information in the internal documents to answer, rather
               than hallucinating from general knowledge.
    """
    question = "What is the capital of France?"
    result   = engine.answer(question)

    answer  = result["answer"].lower()
    context = result["context"]

    assert answer, "Answer must not be empty"

    not_in_docs_phrases = [
        "don't have enough information",
        "do not have enough information",
        "not in the internal documents",
        "no relevant information",
        "not covered",
        "outside the scope",
        "cannot find",
        "no information",
    ]
    has_disclaimer = any(phrase in answer for phrase in not_in_docs_phrases)

    assert has_disclaimer or len(context) == 0, (
        "Expected the model to refuse or indicate lack of information for "
        f"an out-of-domain question.\nGot: {result['answer']}"
    )

    print("\n[TEST 8] Irrelevant question (out-of-domain)")
    print(f"  Q: {question}")
    print(f"  A: {result['answer'][:300]}")
    print(f"  Sources: {[c['source'] for c in context]}")
    print(f"  Context count: {len(context)} (0 means threshold filtered all)")
