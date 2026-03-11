# Proyecto: Agente RAG Telegram con OpenClaw

## Stack Definitivo

- **LLM**: Mistral Nemo 12B (`mistral-nemo:latest` vía Ollama)
- **Embeddings**: `intfloat/multilingual-e5-large` (fallback: `BAAI/bge-m3`)
- **Vector DB**: Qdrant (open-source, Docker)
- **RAG Framework**: LlamaIndex
- **Agent Gateway**: OpenClaw (conecta Telegram ↔ LLM ↔ herramientas)
- **Protocolo**: MCP (Model Context Protocol) para PDF tools y RAG engine
- **Infraestructura**: Docker + Docker Compose (local primero, hostear después)
- **Canal**: Telegram Bot API

## Arquitectura

```
Usuario Telegram
    ↓
OpenClaw (gateway + memoria)
    ↓
Mistral Nemo 12B (Ollama, OpenAI-compatible API)
    ↓ MCP
┌─────────────┐    ┌──────────────┐
│ MCP Server   │    │ MCP Server    │
│ pdf-tools    │    │ rag-engine    │
│              │    │               │
│ PyMuPDF      │    │ LlamaIndex    │
│ pdfplumber   │    │ multilingual  │
│ Tesseract    │    │   -e5-large   │
│              │    │ Qdrant        │
└─────────────┘    └──────────────┘
```

## Servicios Docker Compose

1. **ollama** - Sirve Mistral Nemo 12B (GPU passthrough)
2. **qdrant** - Vector DB persistente
3. **openclaw** - Gateway + Telegram + memoria
4. **mcp-pdf-tools** - Extracción de texto/OCR de PDFs (PyMuPDF + pdfplumber + Tesseract)
5. **mcp-rag-engine** - Pipeline RAG (LlamaIndex + multilingual-e5-large + Qdrant client)

## Flujo Principal

### Ingesta (usuario envía PDF):
1. OpenClaw recibe PDF vía Telegram → guarda en volumen compartido
2. MCP `pdf-tools` → `analyze_document(path)` → extrae texto (PyMuPDF nativo o Tesseract OCR) + tablas (pdfplumber)
3. MCP `rag-engine` → `ingest_document(texto, metadata)` → LlamaIndex chunking (1000 chars, 200 overlap) → embeddings multilingual-e5-large → almacena en Qdrant
4. Bot confirma: "Documento procesado. ¿Qué querés saber?"

### Consulta (usuario hace pregunta):
1. MCP `rag-engine` → `search_documents(query)` → embedding de query → búsqueda semántica Qdrant → top 5 chunks
2. Mistral Nemo genera respuesta con chunks como contexto
3. Respuesta → Telegram

## MCP Tools a implementar

### pdf-tools:
- `extract_text(path)` → texto plano del PDF
- `extract_tables(path)` → tablas en JSON
- `analyze_document(path)` → texto + tablas + metadata completa

### rag-engine:
- `ingest_document(text, metadata)` → chunking + embedding + store en Qdrant
- `search_documents(query, top_k=5)` → búsqueda semántica
- `list_documents()` → documentos ingestados
- `delete_document(doc_id)` → eliminar documento de Qdrant

## Configuración OpenClaw
- Provider: Ollama (http://ollama:11434/v1)
- Modelo: mistral-nemo:latest
- Canal: Telegram (restringido a mi user ID)
- Memoria: MEMORY.md + daily notes + memory flush pre-compactación
- MCP Servers: pdf-tools (SSE) + rag-engine (SSE)

## Variables de entorno (.env)
```
TELEGRAM_BOT_TOKEN=
TELEGRAM_USER_ID=
OLLAMA_HOST=http://ollama:11434
QDRANT_HOST=http://qdrant:6333
EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

## Requisitos locales
- Docker Desktop con GPU support (NVIDIA Container Toolkit)
- GPU con 12GB+ VRAM (RTX 3060/3070/3080/3090/4060/4070/4080/4090)
- 16GB+ RAM sistema
- 20GB+ disco libre

## Notas
- Mistral Nemo 12B necesita ~7GB VRAM en Q4, cabe en cualquier GPU de 12GB+
- multilingual-e5-large corre en CPU sin problemas (~1.2GB RAM)
- Qdrant es liviano, corre sin GPU
- Idiomas: español e inglés
- Tesseract: instalar paquetes spa + eng en el container
- Local first, hostear después (la arquitectura Docker facilita mover a cloud)