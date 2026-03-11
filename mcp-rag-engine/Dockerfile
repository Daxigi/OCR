FROM python:3.11-slim

WORKDIR /app

# Python dependencies
# llama-index-embeddings-huggingface pulls torch; install it first so pip
# can resolve the CUDA-agnostic CPU wheel without extra index flags.
RUN pip install --no-cache-dir "numpy<2.0" && \
    pip install --no-cache-dir \
    mcp \
    uvicorn \
    llama-index \
    llama-index-embeddings-huggingface \
    llama-index-llms-ollama \
    qdrant-client \
    llama-index-vector-stores-qdrant

COPY server.py .

EXPOSE 8002

CMD ["python", "server.py"]
