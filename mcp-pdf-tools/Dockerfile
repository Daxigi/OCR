FROM python:3.11-slim

# System dependencies: Tesseract OCR with Spanish and English language packs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
RUN pip install --no-cache-dir \
    mcp \
    uvicorn \
    pymupdf \
    pdfplumber \
    pillow \
    httpx

COPY server.py .

EXPOSE 8001

CMD ["python", "server.py"]
