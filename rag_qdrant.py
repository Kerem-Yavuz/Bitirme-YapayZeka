#!/usr/bin/env python3
"""
Ders Seçim Chatbot — RAG Engine (Qdrant-Only)

All data stored in Qdrant:
  - ders_secim_docs:    Document vectors + payloads
  - _embedding_cache:   Embedding cache (text hash → vector)
  - _system_config:     System configuration

Zero local file dependencies.

Usage:
  python rag_qdrant.py ingest ./data
  python rag_qdrant.py ask "Bu dersin kredisi kaç?"
  python rag_qdrant.py search "müfredat" --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import re
import sys
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import warnings

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from tqdm import tqdm
import aiohttp

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import config, save_config_to_qdrant


def _apply_e5_prefix(text: str, mode: str) -> str:
    """Add intfloat/e5 instruction prefix when the configured model requires it.

    mode='query'   → prepend 'query: '   (for user questions & router utterances)
    mode='passage' → prepend 'passage: ' (for indexed document chunks)
    """
    if "e5" in config.EMBED_MODEL.lower():
        return f"{mode}: {text}"
    return text


# Optional NLTK
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    # Wrap in a general exception block to catch BadZipFile or other data corruption errors
    try:
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except (LookupError, Exception):
            nltk.download('punkt_tab', quiet=True)
            
        try:
            nltk.data.find('tokenizers/punkt')
        except (LookupError, Exception):
            nltk.download('punkt', quiet=True)
            
        sent_tokenize("Test sentence.")
    except Exception as e:
        print(f"NLTK initialization failed (falling back to simple chunking): {e}")
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """Get CPU, RAM and GPU statistics."""
    info = {
        "cpu_count": os.cpu_count(),
        "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else [0,0,0],
        "torch_device": config.EMBED_DEVICE,
        "cuda_available": torch.cuda.is_available()
    }
    
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
        info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
    
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total"] = f"{vm.total / 1024**3:.2f} GB"
        info["ram_available"] = f"{vm.available / 1024**3:.2f} GB"
        info["ram_percent"] = f"{vm.percent}%"
    except ImportError:
        pass
        
    return info

def log_system_resources():
    """Log current hardware resource usage."""
    info = get_system_info()
    gpu_status = f"GPU: {info.get('gpu_name', 'N/A')} (Allocated: {info.get('gpu_memory_allocated', '0')})" if info['cuda_available'] else "GPU: Not Used (CPU Only)"
    ram_status = f"RAM: {info.get('ram_percent', 'N/A')} used" if 'ram_percent' in info else ""
    logger.info(f"[SYS-RESOURCES] {gpu_status} | {ram_status} | Load: {info['load_avg'][0]}")

# ========================= DATA MODELS =========================

@dataclass
class DocChunk:
    """Represents a document chunk with metadata."""
    doc_id: str
    source: str
    text: str
    checksum: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """Search result with similarity score."""
    chunk: DocChunk
    similarity: float
    rank: int


# ========================= QDRANT EMBEDDING CACHE =========================

class QdrantEmbeddingCache:
    """
    Embedding cache stored in Qdrant _embedding_cache collection.
    Avoids recomputation of embeddings for previously seen texts.
    """

    def __init__(self, client: QdrantClient, dimension: int):
        self.client = client
        self.collection = config.QDRANT_CACHE_COLLECTION
        self.dimension = dimension
        self.hits = 0
        self.misses = 0
        self._available = False
        try:
            self._ensure_collection()
            self._available = True
        except Exception as e:
            logger.warning(f"Embedding cache unavailable (non-fatal): {e}")

    def _ensure_collection(self):
        """Create cache collection if it doesn't exist."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection not in collections:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created cache collection: {self.collection}")
        except Exception as e:
            logger.warning(f"Cache collection setup failed: {e}")

    def _text_to_id(self, text: str) -> int:
        """Convert text to a stable integer ID via hash."""
        h = hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()
        # Use first 15 hex chars → fits in 64-bit int
        return int(h[:15], 16)

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding for text."""
        if not self._available:
            self.misses += 1
            return None
        point_id = self._text_to_id(text)
        try:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=[point_id],
                with_vectors=True,
            )
            if points and points[0].vector:
                self.hits += 1
                return np.array(points[0].vector, dtype=np.float32)
        except Exception:
            pass
        self.misses += 1
        return None

    def set(self, text: str, vector: np.ndarray):
        """Store embedding in cache."""
        if not self._available:
            return
        point_id = self._text_to_id(text)
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={"text_hash": hashlib.sha256(
                            text.encode('utf-8', errors='ignore')
                        ).hexdigest()[:16]},
                    )
                ],
            )
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def set_batch(self, texts: List[str], vectors: np.ndarray):
        """Store multiple embeddings in cache at once."""
        if not self._available:
            return
        points = []
        for text, vec in zip(texts, vectors):
            point_id = self._text_to_id(text)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={"text_hash": hashlib.sha256(
                        text.encode('utf-8', errors='ignore')
                    ).hexdigest()[:16]},
                )
            )
        if points:
            try:
                # Batch upsert in chunks of 100
                for i in range(0, len(points), 100):
                    self.client.upsert(
                        collection_name=self.collection,
                        points=points[i:i + 100],
                    )
            except Exception as e:
                logger.debug(f"Batch cache write failed: {e}")

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
        }


# ========================= SMART CHUNKING =========================

class SmartChunker:
    """Advanced text chunking with sentence awareness."""

    def __init__(self, chunk_size: int = config.CHUNK_SIZE,
                 overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_nltk = NLTK_AVAILABLE

    def chunk_text(self, text: str) -> List[str]:
        """Split text into smart chunks."""
        text = self._clean_text(text)
        if not text or len(text) < config.MIN_CHUNK_SIZE:
            return []

        if self.use_nltk:
            return self._chunk_by_sentences(text)
        else:
            return self._chunk_by_chars(text)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences with overlap."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                overlap_sents = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.overlap:
                        overlap_sents.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break

                current_chunk = overlap_sents + [sentence]
                current_size = sum(len(s) for s in current_chunk)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return [c for c in chunks if len(c) >= config.MIN_CHUNK_SIZE]

    def _chunk_by_chars(self, text: str) -> List[str]:
        """Simple character-based chunking with overlap."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            if len(chunk) >= config.MIN_CHUNK_SIZE:
                chunks.append(chunk)
            start += self.chunk_size - self.overlap

        return chunks


# ========================= FILE PROCESSING =========================

def compute_checksum(text: str) -> str:
    """Compute SHA256 checksum of text."""
    return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()[:16]


def read_file(file_path: Path) -> Optional[str]:
    """Read text from supported file formats."""
    try:
        if file_path.stat().st_size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            logger.warning(f"Skipping large file: {file_path}")
            return None

        suffix = file_path.suffix.lower()

        if suffix in {'.txt', '.md'}:
            return file_path.read_text(encoding='utf-8', errors='ignore')

        elif suffix == '.pdf':
            try:
                reader = PdfReader(str(file_path))
                pages = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                return '\n\n'.join(pages)
            except Exception as e:
                logger.warning(f"PDF read error {file_path}: {e}")
                return None

        return None

    except Exception as e:
        logger.warning(f"File read error {file_path}: {e}")
        return None


def process_single_file(file_path: Path, chunker: SmartChunker,
                        existing_checksums: Dict[str, str]) -> List[DocChunk]:
    """Process a single file into DocChunks."""
    text = read_file(file_path)
    if not text:
        return []

    chunks = chunker.chunk_text(text)
    if not chunks:
        return []

    results = []
    for i, chunk_text in enumerate(chunks):
        doc_id = f"{file_path.stem}::chunk{i}"
        checksum = compute_checksum(chunk_text)

        # Skip if chunk hasn't changed
        if doc_id in existing_checksums and existing_checksums[doc_id] == checksum:
            continue

        chunk = DocChunk(
            doc_id=doc_id,
            source=str(file_path),
            text=chunk_text,
            checksum=checksum,
            chunk_index=i,
            total_chunks=len(chunks),
            metadata={
                'file_size': file_path.stat().st_size,
                'file_modified': file_path.stat().st_mtime,
            }
        )
        results.append(chunk)

    return results


# ========================= VECTOR INDEX (QDRANT) =========================

class VectorIndex:
    """Qdrant-based vector index — all data lives in Qdrant."""

    def __init__(self):
        self.device = config.EMBED_DEVICE
        logger.info(f"Initializing embedding model on {self.device}...")

        self.model = SentenceTransformer(config.EMBED_MODEL, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        self.client = QdrantClient(url=config.QDRANT_URL, timeout=120)
        self.collection_name = config.QDRANT_COLLECTION
        self.cache = QdrantEmbeddingCache(self.client, self.dimension)
        self.chunks: List[DocChunk] = []
        # Thread pool for offloading CPU-bound encode() and blocking Qdrant I/O
        # out of the asyncio event loop (Fix W1)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="embed"
        )

    def _ensure_collection(self):
        """Create document collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.collection_name}")

    def get_existing_checksums(self) -> Dict[str, str]:
        """Get existing doc_id→checksum mapping from Qdrant (for incremental ingest)."""
        checksums = {}
        try:
            # Scroll through all points to get checksums
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["doc_id", "checksum"],
                    with_vectors=False,
                )
                for point in results:
                    if point.payload:
                        checksums[point.payload.get("doc_id", "")] = point.payload.get("checksum", "")
                if offset is None:
                    break
        except Exception:
            pass  # Collection might not exist yet
        return checksums

    def build(self, chunks: List[DocChunk], show_progress: bool = True):
        """Embed chunks and upload to Qdrant."""
        self._ensure_collection()
        self.chunks = chunks

        texts = [c.text for c in chunks]
        # Apply passage prefix for e5-family models before encoding
        texts_to_embed = [_apply_e5_prefix(t, "passage") for t in texts]
        embeddings = self.model.encode(
            texts_to_embed,
            batch_size=config.EMBED_BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Store embeddings in cache (keyed by prefixed text)
        try:
            self.cache.set_batch(texts_to_embed, embeddings)
        except Exception as e:
            logger.warning(f"Cache write failed (non-fatal): {e}")

        # Upload to Qdrant
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(models.PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "doc_id": chunk.doc_id,
                    "checksum": chunk.checksum,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                }
            ))

        # Batch upsert in small chunks with retry
        batch_size = 20
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            for attempt in range(3):
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                    )
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = (attempt + 1) * 5
                        logger.warning(f"Upsert batch {i//batch_size} failed (attempt {attempt+1}/3), retrying in {wait}s: {e}")
                        import time as _time
                        _time.sleep(wait)
                    else:
                        raise RuntimeError(f"Upsert failed after 3 attempts: {e}")

        logger.info(f"{len(points)} chunks uploaded to Qdrant.")

        # Save config snapshot to Qdrant
        try:
            save_config_to_qdrant(self.client)
        except Exception as e:
            logger.warning(f"Config save failed (non-fatal): {e}")

    def _parse_search_results(self, search_results, top_k: int) -> List[SearchResult]:
        """Shared result parsing logic for sync and async search."""
        results = []
        points = search_results.points if hasattr(search_results, 'points') else search_results
        for rank, res in enumerate(points):
            if res.score >= config.SIM_THRESHOLD:
                chunk = DocChunk(
                    doc_id=res.payload.get("doc_id", ""),
                    source=res.payload.get("source", ""),
                    text=res.payload.get("text", ""),
                    checksum=res.payload.get("checksum", ""),
                    chunk_index=res.payload.get("chunk_index", 0),
                    total_chunks=res.payload.get("total_chunks", 0),
                )
                results.append(SearchResult(chunk=chunk, similarity=res.score, rank=rank + 1))
        return results

    def search(self, query: str, top_k: int = config.TOP_K) -> List[SearchResult]:
        """Synchronous vector search — for CLI use only.

        WARNING: Blocks the event loop. Use search_async() inside async contexts.
        """
        prefixed_query = _apply_e5_prefix(query, "query")
        cached = self.cache.get(prefixed_query)
        if cached is not None:
            query_vector = cached.tolist()
        else:
            embed_start = time.time()
            embedding = self.model.encode([prefixed_query], normalize_embeddings=True)[0]
            logger.debug(f"[INDEX-DEBUG] Embedding in {time.time()-embed_start:.3f}s")
            query_vector = embedding.tolist()
            self.cache.set(prefixed_query, embedding)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return self._parse_search_results(search_results, top_k)

    async def search_async(self, query: str, top_k: int = config.TOP_K) -> List[SearchResult]:
        """Non-blocking vector search for use inside async request handlers (Fix W1).

        CPU-bound model.encode() and blocking Qdrant I/O are both executed
        inside a dedicated ThreadPoolExecutor so the asyncio event loop is
        never blocked, allowing concurrent requests to be served.
        """
        loop = asyncio.get_event_loop()
        prefixed_query = _apply_e5_prefix(query, "query")

        # Cache lookup — blocking Qdrant retrieve, run in executor
        cached = await loop.run_in_executor(
            self._executor, self.cache.get, prefixed_query
        )

        if cached is not None:
            logger.debug(f"[INDEX-DEBUG] Cache hit for query: '{query[:30]}...'")
            query_vector = cached.tolist()
        else:
            # model.encode is CPU/GPU-bound — must not block the event loop
            embed_start = time.time()
            embedding = await loop.run_in_executor(
                self._executor,
                lambda: self.model.encode([prefixed_query], normalize_embeddings=True)[0],
            )
            logger.debug(f"[INDEX-DEBUG] Async embed in {time.time()-embed_start:.3f}s")
            query_vector = embedding.tolist()
            # Fire-and-forget cache write (non-critical path)
            loop.run_in_executor(self._executor, self.cache.set, prefixed_query, embedding)

        # Qdrant network I/O — also blocking, run in executor
        search_results = await loop.run_in_executor(
            self._executor,
            lambda: self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            ),
        )
        return self._parse_search_results(search_results, top_k)

    def ensure_ready(self):
        """Ensure Qdrant collection exists and is accessible."""
        self._ensure_collection()

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics from Qdrant."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            }
        except Exception:
            return {"points_count": 0, "vectors_count": 0, "status": "not_found"}


_global_index_instance: Optional[VectorIndex] = None

def get_vector_index() -> VectorIndex:
    """Get or create a global VectorIndex singleton."""
    global _global_index_instance
    if _global_index_instance is None:
        logger.info("Initializing global VectorIndex singleton...")
        log_system_resources()
        _global_index_instance = VectorIndex()
        _global_index_instance.ensure_ready()
        logger.info("VectorIndex singleton ready.")
        log_system_resources()
    return _global_index_instance


# ========================= ASYNC LLM CLIENT =========================

class AsyncLlamaCppClient:
    """Async llama.cpp client using OpenAI-compatible API."""

    def __init__(self, base_url: str = config.LLAMA_CPP_URL):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self._model_name: Optional[str] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=config.LLAMA_CPP_TIMEOUT)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_model(self) -> str:
        """Get model name from server (auto-detect or configured)."""
        if config.LLAMA_CPP_MODEL != "auto":
            return config.LLAMA_CPP_MODEL

        if self._model_name:
            return self._model_name

        try:
            async with self.session.get(f"{self.base_url}/v1/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data') and len(data['data']) > 0:
                        self._model_name = data['data'][0]['id']
                        logger.info(f"Auto-detected model: {self._model_name}")
                        return self._model_name
        except Exception as e:
            logger.warning(f"Failed to auto-detect model: {e}")

        self._model_name = "default"
        return self._model_name

    async def ensure_model(self) -> bool:
        """Check if llama.cpp server is available."""
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            pass
        return False

    async def chat(self, system: str, user: str) -> str:
        """Send chat request using OpenAI-compatible API."""
        model = await self.get_model()

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user}
            ],
            'temperature': config.LLAMA_CPP_TEMPERATURE,
            'max_tokens': config.LLAMA_CPP_MAX_TOKENS,
            'stream': False
        }

        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"llama.cpp API error: {resp.status} - {error_text}")

                data = await resp.json()
                return data['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"llama.cpp chat error: {e}")
            raise

    async def chat_stream(self, system: str, user: str):
        """Stream chat response using OpenAI-compatible API."""
        model = await self.get_model()

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user}
            ],
            'temperature': config.LLAMA_CPP_TEMPERATURE,
            'max_tokens': config.LLAMA_CPP_MAX_TOKENS,
            'stream': True
        }

        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"llama.cpp API error: {resp.status} - {error_text}")

                async for line in resp.content:
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str == "[DONE]": break
                            try:
                                data = json.loads(data_str)
                                token = data["choices"][0]["delta"].get("content", "")
                                if token: yield token
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
        except Exception as e:
            logger.error(f"llama.cpp chat stream error: {e}")
            raise


# ========================= RAG PIPELINE =========================

# Course selection domain system prompt
SYSTEM_PROMPT_RAG = (
    "You are a university course selection assistant. Using the provided CONTEXT and ADDITIONAL INFO, "
    "answer the student's question. Cite your sources using the [doc_id] format. "
    "If the CONTEXT is insufficient, state this and use the ADDITIONAL INFO (Student Profile) and your general knowledge. "
    "Only assist with course selection, curriculum, credits, capacity, and academic topics. "
    "You MUST always answer in the exact same language the user wrote in. "
    "The user's language is indicated at the top of their message as [User language: <code>]. Respect it strictly."
)

SYSTEM_PROMPT_FALLBACK = (
    "You are a university course selection assistant. No CONTEXT was found; "
    "use the ADDITIONAL INFO (Student Profile) if available, and your general knowledge to answer the student. "
    "Only assist with course selection and academic topics. "
    "You MUST always answer in the exact same language the user wrote in. "
    "The user's language is indicated at the top of their message as [User language: <code>]. Respect it strictly."
)


def build_context(results: List[SearchResult],
                  max_tokens: int = config.LLAMA_CPP_MAX_TOKENS) -> str:
    """Build context from search results."""
    # Türkçe agglutinative bir dil — token başına ~3 karakter daha güvenli
    max_chars = (max_tokens * 3) - 2000

    context_parts = []
    total_chars = 0

    for result in results:
        chunk = result.chunk
        snippet = f"[{chunk.doc_id}] (benzerlik: {result.similarity:.3f})\n{chunk.text}\n"

        if total_chars + len(snippet) > max_chars:
            logger.debug(f"[RAG-DEBUG] Context limit reached. Truncating at {total_chars} chars.")
            remaining = max_chars - total_chars
            if remaining > 100:
                context_parts.append(snippet[:remaining] + "...")
            break

        context_parts.append(snippet)
        total_chars += len(snippet)

    return '\n'.join(context_parts)


async def answer_query_async(
    query: str,
    index: VectorIndex,
    top_k: int = config.TOP_K,
    verbose: bool = True,
    llm_url: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Answer query using RAG pipeline (async)."""
    external_context = kwargs.get('external_context')

    # Search for relevant chunks (non-blocking async path)
    search_start = time.time()
    results = await index.search_async(query, top_k=top_k)
    search_time = time.time() - search_start

    if verbose:
        logger.info(f"Search: {search_time:.3f}s, {len(results)} chunks found")

    use_rag = len(results) > 0 and results[0].similarity >= config.SIM_THRESHOLD

    if use_rag:
        context = build_context(results)
        system_prompt = SYSTEM_PROMPT_RAG
        user_prompt = f"Question: {query}\n\nCONTEXT:\n{context}"
    else:
        system_prompt = SYSTEM_PROMPT_FALLBACK
        user_prompt = f"Question: {query}\n\nCONTEXT:\n(empty)"

    if external_context:
        user_prompt += f"\n\nADDITIONAL INFO (Student Profile/Capacity):\n{external_context}"

    user_prompt += "\n\nAnswer:"

    # Get answer from LLM
    base_url = llm_url or config.LLAMA_CPP_URL
    llm_start = time.time()

    async with AsyncLlamaCppClient(base_url=base_url) as client:
        server_available = await client.ensure_model()
        if not server_available:
            logger.warning(f"LLM server not available at {base_url}")
            return {
                'error': 'LLM server not available',
                'query': query,
                'answer': 'LLM sunucusuna bağlanılamıyor. Lütfen sunucunun çalıştığından emin olun.',
                'mode': 'ERROR',
                'results': [],
                'timing': {
                    'search': search_time,
                    'llm': 0,
                    'total': search_time,
                },
            }

        answer = await client.chat(system_prompt, user_prompt)

    llm_time = time.time() - llm_start

    if verbose:
        logger.info(f"LLM response: {llm_time:.3f}s")

    return {
        'query': query,
        'answer': answer,
        'mode': 'RAG' if use_rag else 'FALLBACK',
        'results': [
            {
                'doc_id': r.chunk.doc_id,
                'source': r.chunk.source,
                'similarity': r.similarity,
                'rank': r.rank
            }
            for r in results
        ] if use_rag else [],
        'timing': {
            'search': search_time,
            'llm': llm_time,
            'total': search_time + llm_time
        },
        'best_similarity': results[0].similarity if results else 0.0
    }


async def answer_query_stream(
    query: str,
    index: VectorIndex,
    top_k: int = config.TOP_K,
    llm_url: str = None,
    **kwargs
):
    """Answer query using RAG pipeline (streaming)."""
    external_context = kwargs.get('external_context')
    search_start = time.time()
    results = await index.search_async(query, top_k=top_k)
    search_time = time.time() - search_start

    use_rag = len(results) > 0 and results[0].similarity >= config.SIM_THRESHOLD

    if use_rag:
        context = build_context(results)
        system_prompt = SYSTEM_PROMPT_RAG
        user_prompt = f"Question: {query}\n\nCONTEXT:\n{context}"
    else:
        system_prompt = SYSTEM_PROMPT_FALLBACK
        user_prompt = f"Question: {query}\n\nCONTEXT:\n(empty)"

    if external_context:
        user_prompt += f"\n\nADDITIONAL INFO (Student Profile/Capacity):\n{external_context}"
        logger.debug(f"[RAG-DEBUG] Adding external context of {len(external_context)} chars")

    user_prompt += "\n\nAnswer:"

    base_url = llm_url or config.LLAMA_CPP_URL

    async with AsyncLlamaCppClient(base_url=base_url) as client:
        server_available = await client.ensure_model()
        if not server_available:
            yield json.dumps({"answer": "LLM sunucusuna bağlanılamıyor."}) + "\n"
            return

        async for token in client.chat_stream(system_prompt, user_prompt):
            yield json.dumps({"answer": token}) + "\n"


# ========================= INGEST =========================

def ingest_documents(
    folder_path: Path,
    force_rebuild: bool = False,
    show_progress: bool = True
) -> VectorIndex:
    """Ingest documents and upload to Qdrant."""

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Initialize index
    index = VectorIndex()

    # Get existing checksums from Qdrant (for incremental updates)
    existing_checksums = {}
    if not force_rebuild:
        existing_checksums = index.get_existing_checksums()
        if existing_checksums:
            logger.info(f"Found {len(existing_checksums)} existing chunks in Qdrant")

    if force_rebuild:
        # Delete and recreate collection
        try:
            index.client.delete_collection(index.collection_name)
            logger.info("Deleted existing collection for rebuild")
        except Exception:
            pass

    # Find all supported files
    files = []
    for ext in config.SUPPORTED_EXTENSIONS:
        files.extend(folder_path.rglob(f"*{ext}"))

    if not files:
        raise ValueError(f"No supported files found in {folder_path}")

    logger.info(f"Found {len(files)} files to process")

    # Process files
    chunker = SmartChunker()
    all_chunks = []

    files_iter = tqdm(files, desc="Processing files") if show_progress else files
    for file_path in files_iter:
        try:
            chunks = process_single_file(file_path, chunker, existing_checksums)
            all_chunks.extend(chunks)
            if chunks:
                logger.debug(f"Processed {file_path.name}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"File processing error for {file_path}: {e}")

    if not all_chunks:
        if existing_checksums:
            logger.info("No new/changed documents found. Index is up to date.")
            index.ensure_ready()
            return index
        raise ValueError("No chunks created from files")

    logger.info(f"Created {len(all_chunks)} new chunks from {len(files)} files")

    # Build and upload to Qdrant
    index.build(all_chunks, show_progress=show_progress)

    return index


# ========================= CLI =========================

def cmd_ingest(args):
    """Ingest command."""
    folder = Path(args.folder)
    start_time = time.time()
    index = ingest_documents(folder, force_rebuild=args.force)
    elapsed = time.time() - start_time
    logger.info(f"✓ Indexing completed in {elapsed:.2f}s")
    logger.info(f"✓ Total chunks: {len(index.chunks)}")


def cmd_search(args):
    """Search command (without LLM)."""
    index = VectorIndex()
    index.ensure_ready()

    results = index.search(args.query, top_k=args.top_k)

    if args.json:
        output = {
            'query': args.query,
            'results': [
                {
                    'rank': r.rank,
                    'similarity': r.similarity,
                    'doc_id': r.chunk.doc_id,
                    'source': r.chunk.source,
                    'text': r.chunk.text[:200] + '...' if len(r.chunk.text) > 200 else r.chunk.text
                }
                for r in results
            ]
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*70}")
        print(f"Query: {args.query}")
        print(f"{'='*70}\n")

        for result in results:
            print(f"Rank {result.rank} | Similarity: {result.similarity:.3f}")
            print(f"Source: {result.chunk.source}")
            print(f"Doc ID: {result.chunk.doc_id}")
            print(f"Text: {result.chunk.text[:300]}...")
            print(f"{'-'*70}\n")


def cmd_ask(args):
    """Ask command (with LLM)."""
    index = VectorIndex()
    index.ensure_ready()

    response = asyncio.run(
        answer_query_async(args.question, index, top_k=args.top_k, verbose=not args.json)
    )

    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*70}")
        print(f"Mod: {response['mode']}")
        print(f"Best Similarity: {response['best_similarity']:.3f}")
        print(f"{'='*70}\n")
        print("CEVAP:\n")
        print(response['answer'])

        if response['mode'] == 'RAG':
            print(f"\n{'='*70}")
            print("KAYNAKLAR:\n")
            for r in response['results']:
                print(f"• [{r['doc_id']}] (similarity: {r['similarity']:.3f})")
                print(f"  {r['source']}\n")

        print(f"\n{'='*70}")
        print(f"Timing: Search={response['timing']['search']:.3f}s, "
              f"LLM={response['timing']['llm']:.3f}s, "
              f"Total={response['timing']['total']:.3f}s")


def cmd_info(args):
    """Info command."""
    index = VectorIndex()
    index.ensure_ready()
    info = index.get_collection_info()

    print(f"\n{'='*70}")
    print("QDRANT COLLECTION INFO")
    print(f"{'='*70}\n")
    print(f"Collection: {config.QDRANT_COLLECTION}")
    print(f"Qdrant URL: {config.QDRANT_URL}")
    print(f"Points: {info['points_count']}")
    print(f"Vectors: {info['vectors_count']}")
    print(f"Status: {info['status']}")
    print(f"\nConfig:")
    print(f"  Embed Model: {config.EMBED_MODEL}")
    print(f"  Chunk Size: {config.CHUNK_SIZE}")
    print(f"  Device: {config.EMBED_DEVICE}")
    print(f"  Cache stats: {index.cache.stats()}")


def main():
    parser = argparse.ArgumentParser(
        prog='rag_qdrant.py',
        description='Ders Seçim Chatbot — RAG Engine'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Ingest
    ingest_p = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_p.add_argument('folder', type=str)
    ingest_p.add_argument('--force', action='store_true', help='Force rebuild')

    # Search
    search_p = subparsers.add_parser('search', help='Search without LLM')
    search_p.add_argument('query', type=str)
    search_p.add_argument('--top-k', type=int, default=config.TOP_K)
    search_p.add_argument('--json', action='store_true')

    # Ask
    ask_p = subparsers.add_parser('ask', help='Ask question with LLM')
    ask_p.add_argument('question', type=str)
    ask_p.add_argument('--top-k', type=int, default=config.TOP_K)
    ask_p.add_argument('--json', action='store_true')

    # Info
    subparsers.add_parser('info', help='Show collection info')

    args = parser.parse_args()

    if args.command == 'ingest':
        cmd_ingest(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'ask':
        cmd_ask(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
