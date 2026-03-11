"""
MCP PDF Tools Server
Provides PDF text extraction, table extraction, and document analysis via the MCP protocol.
Transport: SSE on port 8001
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pdf-tools")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ocr_page(page: fitz.Page) -> str:
    """Render a single PDF page to a PNG, then run Tesseract OCR on it."""
    mat = fitz.Matrix(2, 2)  # 2× zoom for better OCR accuracy
    pix = page.get_pixmap(matrix=mat)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        pix.save(tmp_path)

    try:
        result = subprocess.run(
            ["tesseract", tmp_path, "stdout", "-l", "spa+eng", "--psm", "3"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout.strip()
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def extract_text(path: str) -> str:
    """
    Extract plain text from a PDF file.

    Uses PyMuPDF for native text extraction. Falls back to Tesseract OCR
    (Spanish + English) when a page contains no selectable text (e.g. scanned
    documents).

    Args:
        path: Absolute path to the PDF file.

    Returns:
        Extracted text as a single string, with pages separated by form-feeds.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path}")

    doc = fitz.open(str(pdf_path))
    pages_text: list[str] = []

    try:
        for page in doc:
            native_text = page.get_text("text").strip()
            if native_text:
                pages_text.append(native_text)
            else:
                # Scanned page — use OCR
                pages_text.append(_ocr_page(page))
    finally:
        doc.close()

    return "\f".join(pages_text)


@mcp.tool()
def extract_tables(path: str) -> str:
    """
    Extract all tables from a PDF file and return them as JSON.

    Args:
        path: Absolute path to the PDF file.

    Returns:
        JSON string: list of objects with keys "page" (1-based) and "table"
        (list of rows, each row is a list of cell strings).
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    results: list[dict[str, Any]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                # Normalize cells: None → ""
                clean_table = [
                    [cell if cell is not None else "" for cell in row]
                    for row in table
                ]
                results.append({"page": page_num, "table": clean_table})

    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def analyze_document(path: str) -> str:
    """
    Perform a full analysis of a PDF document.

    Combines text extraction and table extraction, and returns structured
    metadata alongside the content.

    Args:
        path: Absolute path to the PDF file.

    Returns:
        JSON string with keys:
          - filename  : base name of the file
          - pages     : total page count
          - word_count: approximate word count of the extracted text
          - text      : full extracted text
          - tables    : list of extracted tables (same format as extract_tables)
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Page count via PyMuPDF (fast, no full parse needed)
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    doc.close()

    text = extract_text(path)
    tables_json = extract_tables(path)
    tables = json.loads(tables_json)

    word_count = len(text.split())

    result = {
        "filename": pdf_path.name,
        "pages": page_count,
        "word_count": word_count,
        "text": text,
        "tables": tables,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8001)
