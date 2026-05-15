#!/usr/bin/env python3
"""
Ders Seçim Chatbot — FastAPI Server

REST API endpoints for the course selection chatbot.
All data stored in Qdrant. Semantic routing is automatic.
Native async streaming — tokens flow to the client in real time.

Usage:
    python qdrant_gui.py
    # Production: uvicorn qdrant_gui:app --host 0.0.0.0 --port 5000 --workers 4
"""

import sys
import time
import logging
import secrets
import os
import random
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from config import config, load_config_from_qdrant, save_config_to_qdrant

# ── Pydantic request models ──────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K
    external_context: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = config.TOP_K

class IngestRequest(BaseModel):
    folder: str
    force: bool = False

# ── Imports ───────────────────────────────────────────────────────────
try:
    from rag_qdrant import (
        VectorIndex,
        ingest_documents,
        answer_query_async,
    )
except ImportError as e:
    print(f"Error importing RAG system: {e}")
    sys.exit(1)

try:
    from router import route_and_answer, get_router
    ROUTER_AVAILABLE = True
except Exception as e:
    ROUTER_AVAILABLE = False
    logging.warning(
        f"Semantic router not available (Error: {e}) "
        "— falling back to direct LLM streaming"
    )

# ── Admin key ─────────────────────────────────────────────────────────
RESET_API_KEY = os.getenv("RESET_API_KEY", "")

# ========================= LIFESPAN =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and graceful shutdown of all singletons."""
    # --- STARTUP ---
    # Stagger workers to avoid simultaneous memory spikes (OOM guard)
    wait_time = random.uniform(0, 10)
    logger.info(f"🚀 API starting up... (stagger delay: {wait_time:.2f}s)")
    await asyncio.sleep(wait_time)

    try:
        from rag_qdrant import get_vector_index
        get_vector_index()                       # Load embedding model + connect Qdrant

        if ROUTER_AVAILABLE:
            from router import get_router, init_http_sessions
            get_router()                         # Build semantic router index
            await init_http_sessions()           # Create persistent LLM HTTP sessions (Fix W2)

        logger.info("✅ Worker ready.")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

    yield  # <-- app is running here

    # --- SHUTDOWN ---
    if ROUTER_AVAILABLE:
        try:
            from router import close_http_sessions
            await close_http_sessions()
        except Exception as e:
            logger.warning(f"Session close error: {e}")

    # BUG-4 FIX: shut down the ThreadPoolExecutor so embed worker threads are joined
    try:
        from rag_qdrant import get_vector_index
        get_vector_index().shutdown()
    except Exception as e:
        logger.warning(f"VectorIndex shutdown error: {e}")

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Worker shut down cleanly.")


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ders Seçim Chatbot API",
    description="RAG-based course selection assistant. All data in Qdrant.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────
index_instance: Optional[VectorIndex] = None
logger = logging.getLogger(__name__)


def get_index() -> VectorIndex:
    """Get the global vector index singleton."""
    from rag_qdrant import get_vector_index
    return get_vector_index()



# ========================= ENDPOINTS =========================

@app.get("/")
async def home():
    """Serve the GUI."""
    gui_path = Path(__file__).parent / "rag_gui.html"
    if gui_path.exists():
        return FileResponse(str(gui_path), media_type="text/html")
    return JSONResponse({"error": "GUI file not found"}, status_code=404)


@app.post("/api/ask")
async def api_ask(body: AskRequest):
    """Ask a question to the AI using semantic routing or direct RAG."""
    start_time = time.time()
    logger.info(f"[API-ASK] New request: '{body.question[:50]}...' (top_k={body.top_k}, has_context={bool(body.external_context)})")
    
    if ROUTER_AVAILABLE:
        from router import route_and_answer_stream

        async def generate():
            async for chunk in route_and_answer_stream(
                body.question, external_context=body.external_context
            ):
                yield chunk

        return StreamingResponse(
            generate(), 
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"}
        )
    else:
        # Fallback: direct RAG pipeline with STREAMING
        from rag_qdrant import answer_query_stream

        try:
            index = get_index()
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Index not available: {e}"
            )

        async def generate_fallback():
            logger.info("[API-ASK] Router unavailable. FALLBACK: Using HARD route (Direct RAG).")
            async for chunk in answer_query_stream(
                body.question, index, top_k=body.top_k, external_context=body.external_context
            ):
                yield chunk
            logger.info(f"[API-ASK] Request completed in {time.time()-start_time:.3f}s")

        return StreamingResponse(generate_fallback(), media_type="text/event-stream")


@app.post("/api/search")
async def api_search(body: SearchRequest):
    """Search for relevant chunks without LLM."""
    try:
        index = get_index()
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Index not available: {e}"
        )

    results = index.search(body.query, top_k=body.top_k)

    return {
        "query": body.query,
        "results": [
            {
                "rank": r.rank,
                "similarity": r.similarity,
                "doc_id": r.chunk.doc_id,
                "source": r.chunk.source,
                "text": r.chunk.text,
            }
            for r in results
        ],
    }

@app.get("/api/status")
async def api_status():
    """Return hardware and system status."""
    from rag_qdrant import get_system_info, get_vector_index
    
    info = get_system_info()
    
    try:
        index = get_vector_index()
        qdrant_info = index.get_collection_info()
    except Exception as e:
        qdrant_info = {"error": str(e)}
        
    return {
        "system": info,
        "qdrant": qdrant_info,
        "config": {
            "model": config.EMBED_MODEL,
            "device": config.EMBED_DEVICE,
            "top_k": config.TOP_K,
            "sim_threshold": config.SIM_THRESHOLD
        },
        "router_available": ROUTER_AVAILABLE
    }


@app.post("/api/ingest")
async def api_ingest(
    files: list[UploadFile] = File(...),
    force: bool = Form(False)
):
    """Ingest documents from uploaded files into Qdrant."""
    import tempfile
    import shutil
    
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        for file in files:
            file_path = temp_dir_path / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
        index = ingest_documents(temp_dir_path, force_rebuild=force, show_progress=False)
        
    elapsed = time.time() - start_time

    # Shutdown the old GUI index instance
    global index_instance
    if index_instance is not None:
        index_instance.shutdown()
    index_instance = index

    # Also update and safely shutdown the global instance in rag_qdrant
    import rag_qdrant
    if rag_qdrant._global_index_instance is not None:
        rag_qdrant._global_index_instance.shutdown()
    rag_qdrant._global_index_instance = index

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "success": True,
        "total_chunks": len(index.chunks),
        "total_files": len(set(c.source for c in index.chunks)),
        "elapsed_time": elapsed,
        "storage": "Qdrant",
    }


@app.get("/api/info")
async def api_info():
    """Get index information from Qdrant."""
    try:
        index = get_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    info = index.get_collection_info()

    return {
        "collection": config.QDRANT_COLLECTION,
        "qdrant_url": config.QDRANT_URL,
        "points_count": info["points_count"],
        "vectors_count": info["vectors_count"],
        "status": info["status"],
        "config": {
            "embed_model": config.EMBED_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "device": config.EMBED_DEVICE,
            "llama_cpp_url": config.LLAMA_CPP_URL,
        },
        "cache_stats": index.cache.stats(),
    }


@app.get("/api/health")
async def health():
    """Health check — reports status of all 3 Qdrant collections."""
    from qdrant_client import QdrantClient

    qdrant_available = False
    collections_status = {}

    try:
        client = QdrantClient(url=config.QDRANT_URL)
        existing = [c.name for c in client.get_collections().collections]
        qdrant_available = True

        for coll in [
            config.QDRANT_COLLECTION,
            config.QDRANT_CONFIG_COLLECTION,
            config.QDRANT_CACHE_COLLECTION,
        ]:
            collections_status[coll] = coll in existing

    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")

    return {
        "status": "healthy" if qdrant_available else "degraded",
        "qdrant_available": qdrant_available,
        "collections": collections_status,
        "router_available": ROUTER_AVAILABLE,
        "storage": "Qdrant (all data)",
    }


@app.post("/api/qdrant/reset")
async def reset_qdrant(request: Request, x_reset_key: Optional[str] = Header(None)):
    """Delete Qdrant collections and start fresh. Requires admin API key."""
    # ── Admin auth guard ──
    if RESET_API_KEY:
        if not x_reset_key or not secrets.compare_digest(x_reset_key, RESET_API_KEY):
            logger.warning(
                "Unauthorized reset attempt from %s", request.client.host
            )
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: geçerli X-Reset-Key header'ı gerekli",
            )
    else:
        logger.warning(
            "RESET_API_KEY not set — /api/qdrant/reset is UNPROTECTED. "
            "Set RESET_API_KEY in .env to secure this endpoint."
        )
    # ── End auth guard ──

    from qdrant_client import QdrantClient

    client = QdrantClient(url=config.QDRANT_URL)

    deleted = []
    for coll in [
        config.QDRANT_COLLECTION,
        config.QDRANT_CONFIG_COLLECTION,
        config.QDRANT_CACHE_COLLECTION,
    ]:
        try:
            client.delete_collection(coll)
            deleted.append(coll)
        except Exception:
            pass

    global index_instance
    if index_instance is not None:
        index_instance.shutdown()
    index_instance = None

    import rag_qdrant
    if rag_qdrant._global_index_instance is not None:
        rag_qdrant._global_index_instance.shutdown()
    rag_qdrant._global_index_instance = None

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "success": True,
        "deleted_collections": deleted,
        "message": "All Qdrant collections deleted",
    }


# ========================= MAIN =========================

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Ders Seçim Chatbot — API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║        Ders Seçim Chatbot — FastAPI Server                ║
╠═══════════════════════════════════════════════════════════╣
║  Server:     http://{args.host}:{args.port}                          ║
║  Docs:       http://{args.host}:{args.port}/docs                     ║
║  Qdrant:     {config.QDRANT_URL:<43}║
║  LLM:        {config.LLAMA_CPP_URL:<43}║
║  Router:     {'✓ Active' if ROUTER_AVAILABLE else '✗ Disabled':<43}║
╠═══════════════════════════════════════════════════════════╣
║  Storage:    Qdrant (all data — zero local files)         ║
╠═══════════════════════════════════════════════════════════╣
║  Endpoints:                                               ║
║    POST /api/ask     — Ask (auto-routed, real-time stream)║
║    POST /api/search  — Search chunks (no LLM)            ║
║    POST /api/ingest  — Ingest documents                  ║
║    GET  /api/info    — Collection info                   ║
║    GET  /api/health  — Health check                      ║
║    POST /api/qdrant/reset — Reset all collections (🔒)   ║
╚═══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "qdrant_gui:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        timeout_keep_alive=300,
    )