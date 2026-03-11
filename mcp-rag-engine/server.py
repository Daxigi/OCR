"""
MCP RAG Engine Server
Provides document ingestion, semantic search, listing, and deletion backed by
Qdrant vector storage and HuggingFace embeddings.
Transport: SSE on port 8002
"""

import json
import os
import uuid
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "http://qdrant:6333")
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
)
COLLECTION_NAME = "documents"

# multilingual-e5-large produces 1024-dimensional vectors
EMBEDDING_DIM = 1024

# ---------------------------------------------------------------------------
# Initialise shared resources at module level so every tool call reuses them
# ---------------------------------------------------------------------------

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
print("Embedding model loaded.")

qdrant_client = QdrantClient(url=QDRANT_HOST)

# Ensure the collection exists
existing = [c.name for c in qdrant_client.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Created Qdrant collection '{COLLECTION_NAME}'.")
else:
    print(f"Qdrant collection '{COLLECTION_NAME}' already exists.")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# SentenceSplitter is stateless – create once and reuse
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("rag-engine")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def ingest_document(text: str, metadata: dict) -> str:
    """
    Chunk, embed, and store a document in Qdrant.

    The document is split with SentenceSplitter (chunk_size=1000,
    chunk_overlap=200), each chunk is embedded with the configured
    HuggingFace model, and all vectors are upserted into the "documents"
    collection.

    Args:
        text    : Full text content of the document.
        metadata: Arbitrary metadata dict (e.g. {"filename": "report.pdf",
                  "source": "/pdfs/report.pdf"}).  A "doc_id" key is
                  auto-generated and added if not present.

    Returns:
        JSON string with keys "doc_id" and "chunks_stored".
    """
    if "doc_id" not in metadata:
        metadata["doc_id"] = str(uuid.uuid4())

    doc_id: str = metadata["doc_id"]

    # Build a LlamaIndex Document and split it into nodes
    document = Document(text=text, metadata=metadata, id_=doc_id)
    nodes = splitter.get_nodes_from_documents([document])

    # Tag every node with the doc_id so we can filter/delete by it later
    for node in nodes:
        node.metadata["doc_id"] = doc_id

    # Build a temporary index that writes directly to our vector_store
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=False,
    )

    return json.dumps({"doc_id": doc_id, "chunks_stored": len(nodes)})


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Perform a semantic search against stored documents.

    Args:
        query : Natural-language query string.
        top_k : Number of top results to return (default 5).

    Returns:
        JSON string: list of objects with keys "score", "text", and
        "metadata" for each matching chunk.
    """
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes_with_scores = retriever.retrieve(query)

    results: list[dict[str, Any]] = []
    for nws in nodes_with_scores:
        results.append(
            {
                "score": round(float(nws.score), 6) if nws.score is not None else None,
                "text": nws.node.get_content(),
                "metadata": nws.node.metadata,
            }
        )

    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def list_documents() -> str:
    """
    List all unique documents stored in Qdrant, based on metadata.

    Returns:
        JSON string: list of objects with keys "doc_id" and any other
        metadata fields present on the first chunk of each document.
    """
    # Scroll through all points and collect unique doc_ids
    offset = None
    seen_doc_ids: set[str] = set()
    documents: list[dict[str, Any]] = []

    while True:
        response, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in response:
            payload = point.payload or {}
            # LlamaIndex stores node metadata under the "_node_content" key
            # as well as directly on the payload depending on the version.
            # Try both locations.
            meta: dict = payload.get("metadata", payload)
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                documents.append(
                    {k: v for k, v in meta.items() if not k.startswith("_")}
                )

        if next_offset is None:
            break
        offset = next_offset

    return json.dumps(documents, ensure_ascii=False, indent=2)


@mcp.tool()
def delete_document(doc_id: str) -> str:
    """
    Delete all chunks belonging to a document from Qdrant.

    Args:
        doc_id: The document ID returned by ingest_document.

    Returns:
        JSON string with key "deleted" set to true and "doc_id".
    """
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        ),
    )
    return json.dumps({"deleted": True, "doc_id": doc_id})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8002)
