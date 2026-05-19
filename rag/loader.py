"""
Document loading and chunking for the DevMind RAG pipeline.

Functions
---------
setup_nltk            -- download required NLTK tokenizer data
extract_text_from_pdf -- extract plain text from a PDF file
load_documents        -- load .txt/.pdf files → (chunks, sources) split by sentence
chunk_by_section      -- alternative chunking that splits on section headers
"""

import os
import re

import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize

SUPPORTED_EXTENSIONS = (".txt", ".pdf")

# Resolve data folder relative to the project root (one level above this package)
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ------------------------------------------------------------------
# NLTK SETUP
# ------------------------------------------------------------------

def setup_nltk() -> None:
    """Download the punkt tokenizer data required by sent_tokenize."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# ------------------------------------------------------------------
# PDF TEXT EXTRACTION
# ------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract plain text from a PDF file using PyMuPDF.

    Performs basic clean-up: collapses runs of whitespace and removes
    isolated page-number lines so they don't pollute chunking.
    """
    doc = fitz.open(file_path)
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    text = "\n".join(pages)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ------------------------------------------------------------------
# DOCUMENT LOADING
# ------------------------------------------------------------------

def load_documents(folder: str = DATA_FOLDER) -> tuple[list[str], list[str]]:
    """
    Load every supported file (.txt, .pdf) from *folder* and split each
    file into sentences.

    Files are processed in sorted order so the index is deterministic across
    runs on the same data set.

    Returns
    -------
    chunks  : list of sentence strings
    sources : parallel list of source filenames (same length as chunks)

    Raises
    ------
    FileNotFoundError  if *folder* does not exist
    ValueError         if no text could be extracted from the folder
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder '{folder}' does not exist. "
            "Create it and add .txt or .pdf files before starting the engine."
        )

    chunks: list[str] = []
    sources: list[str] = []

    for file_name in sorted(os.listdir(folder)):
        file_path = os.path.join(folder, file_name)

        if file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as fh:
                text = fh.read()
        elif file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue

        for sentence in sent_tokenize(text):
            sentence = sentence.strip()
            if sentence:
                chunks.append(sentence)
                sources.append(file_name)

    if not chunks:
        raise ValueError(
            f"No text found. Make sure '{folder}' contains "
            f"supported files ({', '.join(SUPPORTED_EXTENSIONS)}) with content."
        )

    return chunks, sources


# ------------------------------------------------------------------
# SECTION-BASED CHUNKING (alternative to sentence splitting)
# ------------------------------------------------------------------

def chunk_by_section(
    folder: str = DATA_FOLDER,
    min_chunk_len: int = 50,
) -> tuple[list[str], list[str]]:
    """
    Alternative chunker that groups text by section headers instead of
    splitting sentence-by-sentence.

    A line is treated as a header when it starts with '#' or consists
    entirely of uppercase characters (typical ALL-CAPS section titles).

    Returns
    -------
    chunks  : list of section text blocks
    sources : parallel list of source filenames

    Notes
    -----
    Blocks shorter than *min_chunk_len* characters are discarded to avoid
    polluting the index with near-empty sections.
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")

    chunks: list[str] = []
    sources: list[str] = []

    for file_name in sorted(os.listdir(folder)):
        file_path = os.path.join(folder, file_name)

        if file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as fh:
                text = fh.read()
        elif file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue

        current_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            is_header = line.startswith("#") or (
                len(line) > 4 and line.replace(" ", "").isupper()
            )

            if is_header and current_lines:
                block = " ".join(current_lines).strip()
                if len(block) >= min_chunk_len:
                    chunks.append(block)
                    sources.append(file_name)
                current_lines = [line]
            elif line:
                current_lines.append(line)

        # flush the last section
        if current_lines:
            block = " ".join(current_lines).strip()
            if len(block) >= min_chunk_len:
                chunks.append(block)
                sources.append(file_name)

    return chunks, sources
