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

import httpx
import fitz  # PyMuPDF
import pdfplumber
from mcp.server.fastmcp import FastMCP

TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
PDF_DIR = Path("/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

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
# Tool: download from Telegram
# ---------------------------------------------------------------------------


@mcp.tool()
def download_telegram_file(file_id: str) -> str:
    """
    Download a file from Telegram (by file_id) and save it to /pdfs.

    Use this tool when the user sends a document via Telegram. Pass the
    file_id exactly as received. The file is saved to /pdfs/<filename>.

    Args:
        file_id: Telegram file_id of the document.

    Returns:
        JSON string with keys:
          - path     : absolute path where the file was saved (/pdfs/<name>)
          - filename : original filename from Telegram
          - size     : file size in bytes
    """
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in the environment.")

    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    # Step 1: resolve file_id → file_path + file_name
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{base}/getFile", params={"file_id": file_id})
        resp.raise_for_status()
        tg_file = resp.json()

    if not tg_file.get("ok"):
        raise RuntimeError(f"Telegram getFile error: {tg_file}")

    file_info = tg_file["result"]
    tg_path: str = file_info["file_path"]          # e.g. "documents/file_123.pdf"
    file_size: int = file_info.get("file_size", 0)

    # Use the last component of the Telegram path as filename
    filename = Path(tg_path).name
    dest_path = PDF_DIR / filename

    # Step 2: download the actual bytes
    download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{tg_path}"
    with httpx.Client(timeout=120) as client:
        with client.stream("GET", download_url) as stream:
            stream.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in stream.iter_bytes(chunk_size=8192):
                    f.write(chunk)

    return json.dumps(
        {"path": str(dest_path), "filename": filename, "size": file_size},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=8001)
