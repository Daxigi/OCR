FROM python:3.11-slim

WORKDIR /app

# Python dependencies
# llama-index-embeddings-huggingface pulls torch; install it first so pip
# can resolve the CUDA-agnostic CPU wheel without extra index flags.
RUN pip install --no-cache-dir \
    mcp \
    llama-index \
    llama-index-embeddings-huggingface \
    qdrant-client \
    llama-index-vector-stores-qdrant

COPY server.py .

EXPOSE 8002

CMD ["python", "server.py"]
