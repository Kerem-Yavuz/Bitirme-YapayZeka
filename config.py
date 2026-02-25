#!/usr/bin/env python3
"""
Central Configuration Module — Ders Seçim Chatbot

Configuration priority: .env file > Qdrant _system_config > defaults

All other modules import from here. No hardcoded URLs anywhere else.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from datetime import datetime, timezone

import torch

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ========================= DEFAULTS =========================

DEFAULTS = {
    # Qdrant
    "QDRANT_URL": "http://172.18.2.251:30986",
    "QDRANT_COLLECTION": "ders_secim_docs",
    "QDRANT_CONFIG_COLLECTION": "_system_config",
    "QDRANT_CACHE_COLLECTION": "_embedding_cache",

    # Embedding
    "EMBED_MODEL": "all-MiniLM-L6-v2",
    "EMBED_BATCH_SIZE": "128",
    "EMBED_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # Chunking
    "CHUNK_SIZE": "512",
    "CHUNK_OVERLAP": "128",
    "MIN_CHUNK_SIZE": "50",

    # Search
    "TOP_K": "5",
    "SIM_THRESHOLD": "0.25",

    # llama.cpp (main LLM)
    "LLAMA_CPP_URL": "http://yapayzeka.pve.izu.edu.tr",
    "LLAMA_CPP_TIMEOUT": "300",
    "LLAMA_CPP_MAX_TOKENS": "8192",
    "LLAMA_CPP_TEMPERATURE": "0.2",
    "LLAMA_CPP_MODEL": "auto",

    # Router LLM endpoints
    "EASY_LLM_URL": "http://172.18.2.251:31002",
    "HARD_LLM_URL": "http://172.18.2.251:31001",

    # Performance
    "MAX_WORKERS": "4",
    "MAX_FILE_SIZE_MB": "50",

    # File settings
    "SUPPORTED_EXTENSIONS": ".txt,.md,.pdf",
}


def _get(key: str) -> str:
    """Get config value: env var > default."""
    return os.getenv(key, DEFAULTS.get(key, ""))


# ========================= CONFIG CLASS =========================

class Config:
    """Central configuration — all modules import this."""

    # Qdrant
    QDRANT_URL: str = _get("QDRANT_URL")
    QDRANT_COLLECTION: str = _get("QDRANT_COLLECTION")
    QDRANT_CONFIG_COLLECTION: str = _get("QDRANT_CONFIG_COLLECTION")
    QDRANT_CACHE_COLLECTION: str = _get("QDRANT_CACHE_COLLECTION")

    # Embedding
    EMBED_MODEL: str = _get("EMBED_MODEL")
    EMBED_BATCH_SIZE: int = int(_get("EMBED_BATCH_SIZE"))
    EMBED_DEVICE: str = _get("EMBED_DEVICE")

    # Chunking
    CHUNK_SIZE: int = int(_get("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP"))
    MIN_CHUNK_SIZE: int = int(_get("MIN_CHUNK_SIZE"))

    # Search
    TOP_K: int = int(_get("TOP_K"))
    SIM_THRESHOLD: float = float(_get("SIM_THRESHOLD"))

    # llama.cpp
    LLAMA_CPP_URL: str = _get("LLAMA_CPP_URL")
    LLAMA_CPP_TIMEOUT: int = int(_get("LLAMA_CPP_TIMEOUT"))
    LLAMA_CPP_MAX_TOKENS: int = int(_get("LLAMA_CPP_MAX_TOKENS"))
    LLAMA_CPP_TEMPERATURE: float = float(_get("LLAMA_CPP_TEMPERATURE"))
    LLAMA_CPP_MODEL: str = _get("LLAMA_CPP_MODEL")

    # Router LLM endpoints
    EASY_LLM_URL: str = _get("EASY_LLM_URL")
    HARD_LLM_URL: str = _get("HARD_LLM_URL")

    # Performance
    MAX_WORKERS: int = int(_get("MAX_WORKERS"))
    MAX_FILE_SIZE_MB: int = int(_get("MAX_FILE_SIZE_MB"))

    # File settings
    SUPPORTED_EXTENSIONS: set = set(_get("SUPPORTED_EXTENSIONS").split(","))


config = Config()


# ========================= QDRANT CONFIG STORAGE =========================

def save_config_to_qdrant(client=None):
    """Save current config to Qdrant _system_config collection."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    if client is None:
        client = QdrantClient(url=config.QDRANT_URL)

    collection = config.QDRANT_CONFIG_COLLECTION

    # Ensure collection exists (1-dim dummy vector)
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
            shard_number=1,
        )

    payload = {
        "embed_model": config.EMBED_MODEL,
        "embed_batch_size": config.EMBED_BATCH_SIZE,
        "embed_device": config.EMBED_DEVICE,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "min_chunk_size": config.MIN_CHUNK_SIZE,
        "top_k": config.TOP_K,
        "sim_threshold": config.SIM_THRESHOLD,
        "llama_cpp_url": config.LLAMA_CPP_URL,
        "llama_cpp_timeout": config.LLAMA_CPP_TIMEOUT,
        "llama_cpp_max_tokens": config.LLAMA_CPP_MAX_TOKENS,
        "llama_cpp_temperature": config.LLAMA_CPP_TEMPERATURE,
        "llama_cpp_model": config.LLAMA_CPP_MODEL,
        "easy_llm_url": config.EASY_LLM_URL,
        "hard_llm_url": config.HARD_LLM_URL,
        "qdrant_url": config.QDRANT_URL,
        "qdrant_collection": config.QDRANT_COLLECTION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    client.upsert(
        collection_name=collection,
        points=[models.PointStruct(id=1, vector=[0.0], payload=payload)],
    )
    logger.info("Config saved to Qdrant _system_config")


def load_config_from_qdrant(client=None) -> dict | None:
    """Load config from Qdrant _system_config collection. Returns payload dict or None."""
    from qdrant_client import QdrantClient

    if client is None:
        client = QdrantClient(url=config.QDRANT_URL)

    collection = config.QDRANT_CONFIG_COLLECTION

    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection not in collections:
            return None

        points = client.retrieve(collection_name=collection, ids=[1], with_payload=True)
        if points:
            return points[0].payload
    except Exception as e:
        logger.warning(f"Failed to load config from Qdrant: {e}")

    return None
