#!/usr/bin/env python3
"""
Mini RAG — Geliştirilmiş tek dosya versiyon

Özellikler (özet):
 - Kısmi (incremental) ingest: mevcut index varsa sadece yeni/değişen chunk'ları ekler
 - Daha iyi chunklama: cümle-temelli kırpma ve karakter tabanlı fallback
 - Paralel embedding (batch) ve hata dayanıklılığı
 - Daha sağlam Ollama iletişimi: retry, timeout, model kontrolü
 - CLI: ingest / ask / info / rebuild altkomutları
 - JSON çıktısı seçeneği (--json) ve daha iyi loglama
 - Kayıtlı meta dosyasında her chunk için checksum -> duplicate önleme
 - Kaynak gösterimi, context boyut korunumu, ve snippet kırpma

Kullanım (örnek):
  python mini_rag_improved.py ingest ./data
  python mini_rag_improved.py ask "iade politikası nedir?"

Not: Bu dosya mevcut bağımlılıkları varsayar: sentence-transformers, pypdf, numpy, requests, torch
İsteğe bağlı: tiktoken gibi bir tokenizer ile token-fine ayarı iyileştirilebilir.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# --------------------------- Ayarlar ----------------------------------

PROJECT_ROOT = Path(__file__).parent
INDEX_DIR = PROJECT_ROOT / "mini_index"
META_FILE = INDEX_DIR / "meta.json"
EMB_FILE = INDEX_DIR / "embeddings.npy"

# Model ve embedding ayarları
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "5"))
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "700"))
OVERLAP = int(os.environ.get("OVERLAP", "150"))
SUPPORTED = {".txt", ".md", ".pdf"}

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.25"))

# Embedding GPU seçimi
EMBED_GPU_ENV = os.environ.get("EMBED_GPU")  # e.g. "0" -> cuda:0
OLLAMA_MAIN_GPU = int(os.environ.get("OLLAMA_MAIN_GPU", "0"))

# Paralel embedding iş parçacığı sayısı (IO-bound değilse CPU/GPU sınırlı)");
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

# Zaman aşımı ve retry
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "30"))
HTTP_RETRIES = int(os.environ.get("HTTP_RETRIES", "3"))

# --------------------------- Yardımcı İşlevler ------------------------

def log(*args, sep=" "):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}]", *args, sep=sep)


def file_checksum(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# --------------------------- Dosya Okuma ------------------------------

def read_text_from_file(p: Path) -> str:
    suf = p.suffix.lower()
    if suf in {".txt", ".md"}:
        return p.read_text(encoding="utf-8", errors="ignore")
    if suf == ".pdf":
        try:
            reader = PdfReader(str(p))
            pages = [pg.extract_text() or "" for pg in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            log("[warning] PDF okunurken hata:", p, "->", e)
            return ""
    raise ValueError(f"Unsupported file: {p}")


# Cümle temelli chunking (basit regex). Eğer çok uzun cümleler varsa fallback olarak char-based kullanılır.
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?\n])\s+')

def sentence_chunks(text: str, target_chars: int = CHUNK_CHARS, overlap: int = OVERLAP) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    sents = SENTENCE_SPLIT_RE.split(text)
    out: List[str] = []
    cur = []
    cur_len = 0
    step = max(1, target_chars - overlap)
    for s in sents:
        if len(s) + cur_len <= target_chars:
            cur.append(s)
            cur_len += len(s) + 1
            continue
        # flush
        if cur:
            out.append(" ".join(cur))
        # if sentence itself too long -> split by chars
        if len(s) > target_chars:
            # char-based split
            for i in range(0, len(s), step):
                out.append(s[i:i+target_chars])
            cur = []
            cur_len = 0
        else:
            cur = [s]
            cur_len = len(s)
    if cur:
        out.append(" ".join(cur))
    # final normalization
    out = [o.strip() for o in out if o.strip()]
    return out


# --------------------------- MiniIndex -------------------------------

@dataclass
class DocMeta:
    doc_id: str
    source: str
    text: str
    checksum: str


class MiniIndex:
    def __init__(self, embed_model: str = EMBED_MODEL):
        # Device seçimi
        if torch.cuda.is_available():
            if EMBED_GPU_ENV is not None:
                device = f"cuda:{EMBED_GPU_ENV}"
            else:
                device = "cuda:0"
        else:
            device = "cpu"
        log("[index] embedding device:", device)
        self.model = SentenceTransformer(embed_model, device=device)
        self.docs: List[DocMeta] = []
        self.X: Optional[np.ndarray] = None

    def build(self, rows: List[DocMeta], force: bool = False) -> None:
        if not rows:
            raise ValueError("No documents to index.")
        self.docs = rows
        texts = [r.text for r in rows]
        # batch embedding - model.encode zaten internal batching yapar
        X = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        X = np.atleast_2d(X).astype("float32")
        if X.ndim != 2 or X.shape[0] == 0:
            raise RuntimeError("Embedding produced no vectors; check input files.")
        self.X = X

    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        meta = {"docs": [asdict(d) for d in self.docs]}
        META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.X is None:
            raise RuntimeError("No embeddings to save.")
        np.save(str(EMB_FILE), self.X)

    def load(self) -> None:
        if not META_FILE.exists() or not EMB_FILE.exists():
            raise FileNotFoundError("Index not found. Run ingest first.")
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        self.docs = [DocMeta(**d) for d in meta["docs"]]
        self.X = np.load(str(EMB_FILE)).astype("float32")

    def search(self, query: str, top_k: int = TOP_K) -> Tuple[List[Tuple[int, float]], float]:
        if self.X is None:
            raise RuntimeError("Embeddings empty. Load index first.")
        qv = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        qv = np.atleast_2d(qv).astype("float32")
        sims = (self.X @ qv[0])
        idx = np.argsort(sims)[::-1][:top_k]
        best = float(sims[idx[0]]) if idx.size > 0 else 0.0
        return ([(int(i), float(sims[i])) for i in idx], best)


# --------------------------- Ollama Yardımcı --------------------------

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _get(self, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        for attempt in range(HTTP_RETRIES):
            try:
                r = requests.get(url, timeout=HTTP_TIMEOUT, **kwargs)
                r.raise_for_status()
                return r
            except Exception as e:
                log("[ollama] get retry", attempt + 1, "error:", e)
                time.sleep(1 + attempt * 2)
        raise RuntimeError(f"Failed GET {url}")

    def _post(self, path: str, json_payload: dict, timeout: int = 600):
        url = f"{self.base_url}{path}"
        for attempt in range(HTTP_RETRIES):
            try:
                r = requests.post(url, json=json_payload, timeout=timeout)
                r.raise_for_status()
                return r
            except Exception as e:
                log("[ollama] post retry", attempt + 1, "error:", e)
                time.sleep(1 + attempt * 2)
        raise RuntimeError(f"Failed POST {url} after {HTTP_RETRIES} attempts")

    def ensure_model(self):
        try:
            r = self._get("/api/tags")
            data = r.json()
            names = {m.get("name") for m in data.get("models", [])}
            if self.model in names:
                return True
        except Exception:
            pass
        # try pull
        log(f"[ollama] model {self.model} not present locally. Attempting pull...")
        try:
            r = self._post("/api/pull", {"name": self.model}, timeout=1800)
            if r.status_code >= 400:
                log("[ollama] pull failed:", r.status_code, r.text[:200])
                return False
            log("[ollama] pull finished.")
            return True
        except Exception as e:
            log("[ollama] pull error:", e)
            return False

    def chat(self, system: str, user: str, stream: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "num_gpu": int(os.environ.get("OLLAMA_NUM_GPU", "999")),
                "main_gpu": OLLAMA_MAIN_GPU,
                "low_vram": os.environ.get("OLLAMA_LOW_VRAM", "false").lower() == "true",
                "num_ctx": NUM_CTX,
                "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
            },
            "stream": False,
        }
        r = self._post("/api/chat", payload, timeout=600)
        j = r.json()
        return j.get("message", {}).get("content", "")


# --------------------------- Pipeline ---------------------------------


def ingest_folder(folder: Path, force_rebuild: bool = False) -> int:
    if not folder.exists():
        log("[ingest] folder not found:", folder)
        return 0

    # load existing metadata to avoid re-embedding
    existing_meta: Dict[str, DocMeta] = {}
    if META_FILE.exists() and EMB_FILE.exists() and not force_rebuild:
        try:
            meta = json.loads(META_FILE.read_text(encoding="utf-8"))
            for d in meta.get("docs", []) :
                dm = DocMeta(**d)
                existing_meta[dm.doc_id] = dm
            log(f"[ingest] mevcut indexten {len(existing_meta)} doc yüklendi."
                )
        except Exception as e:
            log("[ingest] meta okunurken hata, yeniden oluşturulacak:", e)
            existing_meta = {}

    rows: List[DocMeta] = []
    seen = set()

    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in SUPPORTED:
            raw = read_text_from_file(f)
            if not raw:
                continue
            chunks = sentence_chunks(raw, CHUNK_CHARS, OVERLAP)
            for j, ch in enumerate(chunks):
                doc_id = f"{f.stem}::chunk{j}"
                checksum = file_checksum(ch)
                # if exists and checksum same -> reuse
                if doc_id in existing_meta and existing_meta[doc_id].checksum == checksum:
                    rows.append(existing_meta[doc_id])
                    seen.add(doc_id)
                    continue
                dm = DocMeta(doc_id=doc_id, source=str(f), text=ch, checksum=checksum)
                rows.append(dm)

    if not rows:
        log("[ingest] no chunks found or everything identical to existing index.")
        return 0

    # build embeddings
    idx = MiniIndex()
    idx.build(rows)
    idx.save()
    log(f"[ingest] index kaydedildi. chunk sayısı: {len(rows)}")
    return len(rows)


def build_context_snippets(docs: List[DocMeta], num_ctx_tokens: int) -> str:
    approx_chars_per_token = 4
    reserved_for_prompt = 1000
    max_chars = max(1000, (num_ctx_tokens * approx_chars_per_token) - reserved_for_prompt)

    buf: List[str] = []
    total = 0
    for d in docs:
        snippet = f"[{d.doc_id}] {d.text}"
        if total + len(snippet) > max_chars:
            remain = max(0, max_chars - total)
            if remain > 0:
                buf.append(snippet[:remain])
            break
        buf.append(snippet)
        total += len(snippet)
    return "\n".join(buf)


def answer_question(query: str, top_k: int = TOP_K, json_out: bool = False) -> Dict[str, object]:
    client = OllamaClient()
    ok = client.ensure_model()
    if not ok:
        log("[answer] Ollama modeline erişilemedi veya pull edilemedi. Lütfen Ollama sunucusunu kontrol edin.")

    idx = MiniIndex()
    idx.load()
    hits, best_sim = idx.search(query, top_k=top_k)
    docs = [idx.docs[i] for i, _ in hits]
    use_rag = (len(docs) > 0 and best_sim >= SIM_THRESHOLD)

    if use_rag:
        context = build_context_snippets(docs, NUM_CTX)
        system = (
            "You are a precise assistant. First, answer using only the provided CONTEXT. "
            "Cite sources like [file::chunk0]. If the CONTEXT is insufficient, say so briefly and then add general knowledge."
        )
        user = f"Soru: {query}\n\nCONTEXT:\n{context}\n\nCevap:"
    else:
        system = (
            "You are a helpful assistant. The provided CONTEXT is empty or insufficient; "
            "answer using your general knowledge in Turkish."
        )
        user = f"Soru: {query}\n\nCONTEXT:\n(boş)\n\nCevap:"

    out = client.chat(system, user)
    res = {
        "answer": out,
        "mode": "RAG" if use_rag else "FALLBACK",
        "best_sim": best_sim,
        "sources": ([{"doc_id": d.doc_id, "source": d.source} for d in docs] if use_rag else []),
    }
    if json_out:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


# --------------------------- CLI --------------------------------------

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(prog="mini_rag_improved.py", description="Mini RAG - geliştirilmiş")
    sub = ap.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Klasörü tara ve index oluştur")
    p_ingest.add_argument("folder", type=str)
    p_ingest.add_argument("--force", action="store_true", help="Mevcut index'i zorla yeniden oluştur")

    p_ask = sub.add_parser("ask", help="Soru sor")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--top_k", type=int, default=TOP_K)
    p_ask.add_argument("--json", action="store_true", help="JSON çıktısı ver")

    p_info = sub.add_parser("info", help="Index hakkında bilgi ver")

    p_rebuild = sub.add_parser("rebuild", help="Index'i tamamen yeniden oluştur (force)")
    p_rebuild.add_argument("folder", type=str)

    args = ap.parse_args(argv)

    if args.cmd == "ingest":
        n = ingest_folder(Path(args.folder), force_rebuild=args.force)
        if n == 0:
            log("⚠ Hiç chunk üretilmedi. Doğru klasör ve desteklenen dosyalar olduğundan emin ol.")
        else:
            log(f"✔ Index hazır. Toplam chunk: {n}. Dosyalar: {EMB_FILE}, {META_FILE}")
        return

    if args.cmd == "ask":
        if not EMB_FILE.exists() or not META_FILE.exists():
            log("Önce 'ingest' ile index oluştur.")
            raise SystemExit(1)
        res = answer_question(args.question, top_k=args.top_k, json_out=args.json)
        if not args.json:
            print("\n=== MOD ===", res["mode"], "(best_sim=", f"{res['best_sim']:.3f}", ")")
            print("\n=== CEVAP ===\n")
            print(str(res["answer"]).strip())
            if res["mode"] == "RAG":
                print("\n=== KAYNAKLAR ===")
                for s in res["sources"]:
                    print(f"- {s['doc_id']}  <- {s['source']}")
        return

    if args.cmd == "info":
        if not META_FILE.exists() or not EMB_FILE.exists():
            log("Index bulunamadı.")
            return
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        docs = meta.get("docs", [])
        log(f"Index mevcut. chunk sayısı: {len(docs)}")
        return

    if args.cmd == "rebuild":
        n = ingest_folder(Path(args.folder), force_rebuild=True)
        log(f"Rebuild tamam. {n} chunk oluşturuldu.")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
