# Proyecto: Agente RAG Telegram con OpenClaw

## Concepto Central

Agente de Telegram que construye una **base de conocimiento incremental** a partir de PDFs. Cada documento que se ingesta se suma al corpus total en Qdrant. Las preguntas del usuario se responden buscando en **la totalidad de los documentos ingestados**, no en un PDF individual. Es una memoria acumulativa que crece con cada documento.

## Stack

- **LLM**: Mistral Nemo 12B (`mistral-nemo:latest` vía Ollama)
- **Embeddings**: `intfloat/multilingual-e5-large` (fallback: `BAAI/bge-m3`)
- **Vector DB**: Qdrant (colección `documents`, vectores 1024 dims, cosine similarity)
- **RAG Framework**: LlamaIndex
- **Agent Gateway**: OpenClaw
- **Protocolo**: MCP (Model Context Protocol)
- **Infra**: Docker Compose (5 contenedores)
- **Canal**: Telegram Bot API

## Los 5 Contenedores (estado actual)

### 1. ollama (LLM) ✅ Construido
- Mistral Nemo 12B con GPU passthrough NVIDIA
- API OpenAI-compatible en `http://ollama:11434/v1`
- Volumen: `ollama_data`

### 2. qdrant (Vector DB) ✅ Construido
- Colección `documents`, vectores 1024 dims, cosine similarity
- Puertos: 6333 (HTTP), 6334 (gRPC)
- Volumen: `qdrant_data`

### 3. mcp-pdf-tools (puerto 8001) ✅ Construido
MCP server con 3 tools:
- `extract_text(path)` → PyMuPDF nativo, Tesseract OCR fallback (esp + eng)
- `extract_tables(path)` → pdfplumber, JSON por página
- `analyze_document(path)` → combina ambos + metadata (nombre, páginas, word count)

### 4. mcp-rag-engine (puerto 8002) ✅ Construido
MCP server con 4 tools:
- `ingest_document(text, metadata)` → chunks (1000 chars, 200 overlap) → embeddings multilingual-e5-large → Qdrant
- `search_documents(query, top_k=5)` → embedding de query → búsqueda semántica en TODO el corpus → top K chunks con score
- `list_documents()` → lista docs únicos en la BD
- `delete_document(doc_id)` → elimina chunks de un documento

### 5. openclaw (gateway, puerto 8080) ✅ Construido
- Conecta Telegram ↔ Mistral ↔ MCP servers
- Restringido a un user ID de Telegram
- System prompt con instrucciones de uso de herramientas
- Flujo: PDF → extrae → ingesta → confirma / Pregunta → busca en TODO el corpus → responde

## Flujo de Ingesta (incremental)

```
PDF 1 → extract → chunk → embed → Qdrant [doc1_chunk1, doc1_chunk2, ...]
PDF 2 → extract → chunk → embed → Qdrant [doc1_chunk1, ..., doc2_chunk1, doc2_chunk2, ...]
PDF N → extract → chunk → embed → Qdrant [todos los chunks de todos los docs]
```

Cada documento se identifica por metadata (nombre, fecha de ingesta, páginas). Los chunks se acumulan. La búsqueda semántica recorre TODOS los chunks de TODOS los documentos y devuelve los más relevantes independientemente de qué PDF vinieron.

## Flujo de Consulta (cross-document)

```
Usuario: "¿Qué dice sobre las cláusulas de terminación?"
    → search_documents("cláusulas de terminación", top_k=5)
    → Qdrant devuelve 5 chunks (pueden ser de distintos PDFs)
    → Cada chunk incluye metadata: de qué documento viene
    → Mistral genera respuesta citando las fuentes:
      "Según [Contrato_A.pdf], ... Además, en [Contrato_B.pdf]..."
```

## Lo que FALTA implementar

| # | Pendiente | Detalle |
|---|-----------|---------|
| 1 | **Recepción de PDF de Telegram** | OpenClaw debe interceptar el archivo que envía el usuario por Telegram, descargarlo y guardarlo en `/pdfs` (volumen compartido). Actualmente no está implementada esta lógica. |
| 2 | **Validación del config.yaml de OpenClaw** | Verificar que el config.yaml es compatible con la versión real de OpenClaw/AgentGateway instalada. |
| 3 | **Pull automático del modelo** | `mistral-nemo:latest` se debe bajar manualmente con `docker exec ollama ollama pull mistral-nemo`. Automatizar en el entrypoint o init container. |
| 4 | **Health checks entre servicios** | No hay health checks. Los servicios pueden intentar conectarse antes de que los demás estén listos. |

## Variables de entorno (.env)
```
TELEGRAM_BOT_TOKEN=
TELEGRAM_USER_ID=
OLLAMA_HOST=http://ollama:11434
QDRANT_HOST=http://qdrant:6333
EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

## Notas técnicas
- Mistral Nemo 12B: ~7GB VRAM (Q4), cabe en GPU 12GB+
- multilingual-e5-large: corre en CPU (~1.2GB RAM)
- Qdrant: liviano, sin GPU
- Idiomas: español + inglés
- Local primero, hostear después (Docker facilita migración)