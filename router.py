#!/usr/bin/env python3
"""
Semantic Router for RAG — Routes queries to Easy LLM, Hard LLM, or Reject.
"""

import asyncio
import time
import logging
import json
import hashlib
from typing import Dict, Any, AsyncGenerator

from semantic_router import Route, SemanticRouter
try:
    from semantic_router.encoders import BaseEncoder
except ImportError:
    from semantic_router.encoders.base import BaseEncoder
from semantic_router.index.local import LocalIndex

import aiohttp
import cachetools
from config import config

# Language detection (multilingual support)
try:
    from langdetect import detect as _langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


def detect_language(text: str) -> str:
    """Detect the language of the input text.

    Returns an ISO 639-1 code (e.g. 'tr', 'en', 'fr', 'de', 'ar').
    Falls back to 'unknown' if langdetect is unavailable or the text is too short.
    """
    if not LANGDETECT_AVAILABLE or len(text.strip()) < 5:
        return "unknown"
    try:
        return _langdetect(text)
    except Exception:
        return "unknown"

# ========================= L1 TTL CONTEXT CACHE (Fix P4) =========================
# Caches (route, context) for repeated identical queries — skips embed+search.
# TTL=300s: stale after 5 min, safe for a university chatbot workload.
_context_cache: cachetools.TTLCache = cachetools.TTLCache(maxsize=256, ttl=300)

# ========================= PERSISTENT HTTP SESSION POOL (Fix W2) =========================
# One aiohttp.ClientSession per LLM endpoint, created at startup and reused
# across all requests. Eliminates per-request TCP handshake overhead.
_http_sessions: Dict[str, aiohttp.ClientSession] = {}


async def init_http_sessions():
    """Create persistent aiohttp sessions for each LLM endpoint.
    Call once from the FastAPI lifespan startup.
    """
    timeout = aiohttp.ClientTimeout(total=config.LLAMA_CPP_TIMEOUT)
    for key, url in {
        "easy": config.EASY_LLM_URL,
        "hard": config.HARD_LLM_URL,
    }.items():
        _http_sessions[key] = aiohttp.ClientSession(
            base_url=url, timeout=timeout
        )
    logger.info(f"✅ Persistent HTTP sessions created for easy/hard LLM endpoints.")


async def close_http_sessions():
    """Close all persistent sessions. Call from FastAPI lifespan shutdown."""
    for session in _http_sessions.values():
        await session.close()
    _http_sessions.clear()
    logger.info("HTTP sessions closed.")


def _apply_query_prefix(text: str) -> str:
    """Apply 'query: ' prefix for e5-family models."""
    if "e5" in config.EMBED_MODEL.lower():
        return f"query: {text}"
    return text

# Note: We use HuggingFaceEncoder directly because using multiple workers (e.g., 4) 
# would trigger the model to be loaded 4 times into memory, causing OOM errors 
# or unnecessary overhead. Keeping it single-worker/local ensures a single load.

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ========================= CONFIGURATION =========================

REJECT_MESSAGE = (
    "Üzgünüm, bu soru kapsamım dışında. "
    "Sadece ders seçimi, müfredat ve akademik konularda yardımcı olabilirim."
)

# ========================= ROUTE DEFINITIONS =========================

easy_route = Route(
    name="easy",
    utterances=[
        "Bu dersin kredisi kaç?", "Dersin saati ne zaman?", "Dersin kontenjanı ne kadar?",
        "Sınav tarihi ne zaman?", "Ders saatleri nelerdir?", "Ödev teslim tarihi ne?",
        "Devam zorunluluğu var mı?", "Dersin hocası kim?", "Ders hangi gün?",
        "Dersin ön koşulu ne?", "Kontenjan doldu mu?", "Hangi dersler seçmeli?",
        "Dersin dili ne?", "Bu ders online mı?", "Dersin kodu ne?",
        "Bu ders kaçıncı sınıf dersi?", "Vize tarihi ne zaman?", "Final sınavı ne zaman?",
        "Dersin AKTS değeri kaç?", "Bu ders zorunlu mu?", "Ders hangi bölüme ait?",
        "Dersin sınıfı nerede?", "Ders hangi fakülteye bağlı?", "Bu dersi hangi hoca veriyor?",
        "Bu dersin bütünleme sınavı var mı?", "Kayıt yenileme tarihi ne zaman?",
        "Dersin başlangıç tarihi ne?", "Bu dersi kimler alabilir?", "Proje teslim tarihi ne?",
        "Labın saati ne zaman?", "Dersin kitabı ne?", "Bu dönem açılan dersler hangileri?",
        "Sınav yeri neresi?", "Not ortalaması nasıl hesaplanır?", "Transkript nasıl alınır?",
        "What is the exam schedule?", "When does registration end?", "What are the course prerequisites?",
        "How many credits is this course?", "What is the grading policy?", "What is the attendance policy?",
        "When is the homework deadline?", "Who is the instructor for this course?", "What time is the lecture?",
        "Where is the classroom?", "Is this course mandatory or elective?", "What is the course code?",
        "When is the final exam?", "When is the midterm exam?", "What textbook does this course use?",
        "How is the grade calculated?", "Is there a lab session for this course?", "What are the office hours?",
        "How do I enroll in this course?", "What is the course capacity?",
        "hoca kim", "vize ne zaman", "kredi kaç", "CS101 syllabus", "ders saatleri", "kontenjan",
        "devam zorunluluğu", "BBM401 önkoşul",
    ],
)

hard_route = Route(
    name="hard",
    utterances=[
        "Bu dersleri alırsam müfredata uyar mı?", "Hangi seçmeli dersler daha uygun olur?",
        "Bu iki dersi aynı dönem alabilir miyim?", "Ders programımı nasıl optimize edebilirim?",
        "Bu dersler arasındaki farkları açıklar mısın?", "Mezuniyet için hangi dersleri almalıyım?",
        "Ders yükümü nasıl dengeleyebilirim?", "Bu bölümün müfredatını analiz et",
        "Seçmeli derslerin avantaj ve dezavantajlarını karşılaştır", "Bu ders planıyla kaç dönemde mezun olurum?",
        "Bu derslerin birbirleriyle ilişkisini açıkla", "Çift anadal yapmak istiyorum, ders planımı nasıl ayarlamalıyım?",
        "Yandal programına uygun ders seçimi nasıl olmalı?", "Bu dönem hangi dersleri almam daha mantıklı?",
        "Staj dönemimde hangi dersleri alabilirim?", "Mezuniyetimi uzatmadan hangi dersleri ekleyebilirim?",
        "Bu dersler kariyer hedeflerime uygun mu?", "Ders çakışmalarını nasıl çözebilirim?",
        "Tekrar almam gereken derslerle programımı nasıl düzenlemeliyim?", "Erasmus döneminde hangi dersleri saydırabilirim?",
        "Üst dönemden ders almak için ne yapmalıyım?", "Bu derslerin iş hayatına katkısı ne olur?",
        "Bölüm değiştirirsem hangi derslerim sayılır?", "GPA'mı yükseltmek için hangi dersleri seçmeliyim?",
        "Bu ders planıyla yaz okuluna ihtiyacım olur mu?", "Zorunlu ve seçmeli dersleri nasıl dengeleyebilirim?",
        "Bu alandaki dersler arasında en faydalısı hangisi?", "Lisansüstüne hazırlık için hangi dersleri önerirsin?",
        "Compare the advantages of these elective courses", "Analyze my course plan for graduation requirements",
        "Explain the differences between these two courses", "Help me plan my semester schedule",
        "What are the trade-offs between these courses?", "Which electives would be best for my career goals?",
        "Can I take these two courses in the same semester?", "How should I plan my courses for a double major?",
        "What is the best strategy to graduate on time?", "How do these courses relate to each other?",
        "Which courses should I prioritize this semester?", "Recommend a balanced course load for next semester",
        "What courses can I take during my Erasmus exchange?", "How will dropping this course affect my graduation?",
        "What electives complement my major the best?", "Should I take summer school or overload next semester?",
        "Analyze the difficulty level of this course combination", "How can I improve my GPA with course selection?",
        "Eğer bu dersi geçemezsem seneye hangi dersleri alamam?", "Dönem uzatmamak için hangi dersi bırakmalıyım?",
        "Bu iki seçmeli ders arasında kalsam hangisi GPA için daha iyi?", "If I fail this course, what courses can't I take next year?",
    ],
)

# ========================= ROUTER SETUP =========================

_router_instance = None
_index_instance = None


def get_rag_index():
    """Return the global VectorIndex singleton."""
    from rag_qdrant import get_vector_index
    return get_vector_index()


def get_router() -> SemanticRouter:
    global _router_instance
    if _router_instance is None:
        from rag_qdrant import get_vector_index
        index_instance = get_vector_index() # Ensure model is loaded
        
        logger.info(f"Initializing SemanticRouter (Memory Optimized)...")
        
        # BUG-7 FIX: Custom encoder to securely use the shared model instance 
        # without Pydantic triggering a duplicate load into RAM.
        class SharedEncoder(BaseEncoder):
            name: str
            type: str = "huggingface"

            def __init__(self, name: str, shared_model):
                super().__init__(name=name)
                object.__setattr__(self, "_shared_model", shared_model)

            def __call__(self, docs: list[str]) -> list[list[float]]:
                embeddings = self._shared_model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.tolist()

        encoder = SharedEncoder(name=config.EMBED_MODEL, shared_model=index_instance.model)
        logger.info("✅ Router is using the shared model instance safely.")

        index = LocalIndex()
        _router_instance = SemanticRouter(
            encoder=encoder, 
            routes=[easy_route, hard_route],
            index=index
        )
        
        # MANUALLY POPULATE INDEX: This ensures the index is ready immediately
        try:
            logger.info("Manually indexing routes for SemanticRouter...")
            all_utterances = []
            all_route_names = []
            for r in [easy_route, hard_route]:
                for u in r.utterances:
                    all_utterances.append(u)
                    all_route_names.append(r.name)

            # Apply e5 query prefix before encoding utterances so the
            # router embedding space matches search_async() query embeddings.
            utterances_to_encode = [_apply_query_prefix(u) for u in all_utterances]
            utterance_embeddings = encoder(utterances_to_encode)
            
            # Add to index manually
            _router_instance.index.add(
                embeddings=utterance_embeddings, 
                routes=all_route_names,
                utterances=all_utterances
            )
            _router_instance.index.ready = True
            logger.info("✅ SemanticRouter index manually built and ready!")
        except Exception as e:
            logger.warning(f"Manual indexing failed: {e}")
            
    return _router_instance

# ========================= LLM CLIENTS =========================

async def call_llm_stream(route_name: str, system: str, user: str) -> AsyncGenerator[str, None]:
    """Stream chat response using the persistent session for the given route."""
    session = _http_sessions.get(route_name)
    # Fallback: create a temporary session if pool not yet initialised
    if session is None:
        url = config.EASY_LLM_URL if route_name == "easy" else config.HARD_LLM_URL
        timeout = aiohttp.ClientTimeout(total=config.LLAMA_CPP_TIMEOUT)
        session = aiohttp.ClientSession(base_url=url, timeout=timeout)
        owns_session = True
    else:
        owns_session = False

    payload = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": config.LLAMA_CPP_TEMPERATURE,
        "max_tokens": config.LLAMA_CPP_MAX_TOKENS,
        "stream": True,
    }
    try:
        # BUG-6 FIX: session already has base_url configured, so we use relative path
        async with session.post("/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"LLM error: {resp.status}")
            buffer = ""
            first_token_time = None
            async for chunk in resp.content.iter_any():
                buffer += chunk.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        token = data["choices"][0]["delta"].get("content", "")
                        if token:
                            if first_token_time is None:
                                first_token_time = time.time()
                                logger.info(f"[LLM-DEBUG] First token. Content: '{token[:20]}'")
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.debug(f"JSON decode error: {e}")
    finally:
        if owns_session:
            await session.close()


async def call_llm(route_name: str, system: str, user: str) -> str:
    """Non-streaming LLM call using the persistent session for the given route."""
    session = _http_sessions.get(route_name)
    owns_session = False
    if session is None:
        url = config.EASY_LLM_URL if route_name == "easy" else config.HARD_LLM_URL
        timeout = aiohttp.ClientTimeout(total=config.LLAMA_CPP_TIMEOUT)
        session = aiohttp.ClientSession(base_url=url, timeout=timeout)
        owns_session = True

    payload = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": config.LLAMA_CPP_TEMPERATURE,
        "max_tokens": config.LLAMA_CPP_MAX_TOKENS,
        "stream": False,
    }
    try:
        # BUG-6 FIX: session already has base_url configured, so we use relative path
        async with session.post("/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"LLM error: {resp.status}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    finally:
        if owns_session:
            await session.close()

# ========================= RAG INTEGRATION =========================

async def get_rag_context_async(query: str, top_k: int = 5) -> str:
    """Retrieve RAG context for a query, with L1 TTL cache (Fix P4).

    Cache key is an MD5 of (query, top_k) so identical queries skip
    the embed+search pipeline entirely for up to 5 minutes.
    """
    cache_key = hashlib.md5(f"{query}|{top_k}".encode()).hexdigest()
    if cache_key in _context_cache:
        logger.info(f"[RAG-CACHE] L1 hit for query: '{query[:40]}'")
        return _context_cache[cache_key]

    start = time.time()
    try:
        from rag_qdrant import build_context
        index = get_rag_index()   # singleton — model not reloaded
        search_start = time.time()
        results = await index.search_async(query, top_k=top_k)  # non-blocking
        search_duration = time.time() - search_start

        context = build_context(results) if results else "(bağlam bulunamadı)"
        logger.info(
            f"[RAG-DEBUG] Search {search_duration:.3f}s. "
            f"Results: {len(results)}. Context: {len(context)} chars"
        )
        _context_cache[cache_key] = context
        return context
    except Exception as e:
        logger.warning(f"[RAG-ERROR] Context error after {time.time()-start:.3f}s: {e}")
        return "(bağlam hatası)"

# ========================= MAIN ROUTING LOGIC =========================

def build_prompts(query: str, context: str, external_context: str = None):
    """Build system + user prompt pair.

    Detects the user's language and injects it as an explicit tag so the LLM
    always responds in the correct language regardless of context language.
    """
    lang = detect_language(query)
    system = (
        "You are a university course selection assistant. Using the provided CONTEXT and ADDITIONAL INFO, "
        "answer the student's question. Cite your sources. "
        "If the CONTEXT is insufficient, state this and use the ADDITIONAL INFO (Student Profile) and your general knowledge. "
        "Only assist with course selection and academic topics. "
        "You MUST always answer in the exact same language the user wrote in. "
        "The user's language is indicated at the top of their message as [User language: <code>]. Respect it strictly."
    )
    user = f"[User language: {lang}]\nQuestion: {query}\n\nCONTEXT:\n{context}"
    if external_context:
        user += f"\n\nADDITIONAL INFO (Student Profile/Capacity):\n{external_context}"
    user += "\n\nAnswer:"
    return system, user

async def route_and_answer_stream(query: str, external_context: str = None):
    start_time = time.time()
    router = get_router()
    route_start = time.time()
    try:
        # Apply e5 prefix to the user query before routing for consistency
        route_result = router(_apply_query_prefix(query))
        route_name = route_result.name if route_result else None
    except Exception as e:
        logger.error(f"[ROUTER-ERROR] Router failed: {e}. Falling back to 'hard' route.")
        route_name = "hard"

    route_duration = time.time() - route_start
    logger.info(f"[ROUTER-DEBUG] Query: '{query[:50]}' | Route: {route_name} | Duration: {route_duration:.3f}s")

    if route_name is None:
        logger.warning(f"[ROUTER-DEBUG] Query rejected by semantic router.")
        yield json.dumps({"answer": REJECT_MESSAGE, "status": "done"}) + "\n"
        return

    # Async context retrieval — non-blocking, L1 cache applied inside
    context = await get_rag_context_async(query)
    system, user = build_prompts(query, context, external_context)

    logger.info(
        f"[LLM-DEBUG] Calling LLM ({route_name}). "
        f"System: {len(system)} chars. User: {len(user)} chars."
    )

    try:
        token_count = 0
        first_token_time = None
        async for token in call_llm_stream(route_name, system, user):
            if first_token_time is None:
                first_token_time = time.time()
                logger.info(f"[LLM-DEBUG] First token in {first_token_time - start_time:.3f}s")
            token_count += 1
            yield json.dumps({"answer": token}) + "\n"

        total_duration = time.time() - start_time
        logger.info(f"[LLM-DEBUG] Stream done. Tokens: ~{token_count}. Total: {total_duration:.3f}s")
    except Exception as e:
        logger.error(f"[LLM-ERROR] Streaming failed after {time.time()-start_time:.3f}s: {e}")
        yield json.dumps({"answer": f"\n[Hata: {str(e)}]"}) + "\n"


async def route_and_answer(query: str, external_context: str = None) -> Dict[str, Any]:
    """Non-streaming version for CLI or testing."""
    start = time.time()
    router = get_router()
    route_result = router(_apply_query_prefix(query))
    route_name = route_result.name if route_result else None

    if route_name is None:
        return {"query": query, "answer": REJECT_MESSAGE, "route": "rejected", "timing": {"total": time.time()-start}}

    context = await get_rag_context_async(query)
    system, user = build_prompts(query, context, external_context)

    try:
        answer = await call_llm(route_name, system, user)
    except Exception as e:
        answer = f"Hata oluştu: {str(e)}"

    return {"query": query, "answer": answer, "route": route_name, "timing": {"total": time.time()-start}}

# ========================= CLI =========================

def cmd_ask(args):
    result = asyncio.run(route_and_answer(args.question))
    print(f"\nAnswer: {result['answer']}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    args = parser.parse_args()
    if args.command == "ask":
        cmd_ask(args)