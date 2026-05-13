#!/usr/bin/env python3
"""
Semantic Router for RAG — Routes queries to Easy LLM, Hard LLM, or Reject.
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any, AsyncGenerator

from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.index.local import LocalIndex

import aiohttp
from config import config

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
        
        # Initialize standard encoder
        encoder = HuggingFaceEncoder(name=config.EMBED_MODEL)
        
        # MONKEY-PATCH: Bypass Pydantic validation to force model sharing
        try:
            object.__setattr__(encoder, 'model', index_instance.model)
            logger.info("✅ Router is using the shared model instance (Forced).")
        except Exception as e:
            logger.warning(f"Shared model injection failed: {e}")

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
            
            # Use our encoder to get embeddings for all utterances
            utterance_embeddings = encoder(all_utterances)
            
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

async def call_llm_stream(base_url: str, system: str, user: str) -> AsyncGenerator[str, None]:
    """Stream chat response from a llama.cpp server."""
    timeout = aiohttp.ClientTimeout(total=config.LLAMA_CPP_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": config.LLAMA_CPP_TEMPERATURE,
            "max_tokens": config.LLAMA_CPP_MAX_TOKENS,
            "stream": True,
        }
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
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
                            # Log first token content to debug 'thinking'
                            if first_token_time is None:
                                first_token_time = time.time()
                                snippet = token.replace("\n", "\\n")[:20]
                                logger.info(f"[LLM-DEBUG] First token received. Content: '{snippet}'")
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.debug(f"JSON decode error: {e}")
                        continue

async def call_llm(base_url: str, system: str, user: str) -> str:
    """Send chat request to a llama.cpp server and get full response."""
    timeout = aiohttp.ClientTimeout(total=config.LLAMA_CPP_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": config.LLAMA_CPP_TEMPERATURE,
            "max_tokens": config.LLAMA_CPP_MAX_TOKENS,
            "stream": False,
        }
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"LLM error: {resp.status}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

# ========================= RAG INTEGRATION =========================

def get_rag_context(query: str, top_k: int = 5) -> str:
    start = time.time()
    try:
        from rag_qdrant import build_context
        index = get_rag_index()   # singleton — model yüklenmez tekrar
        search_start = time.time()
        results = index.search(query, top_k=top_k)
        search_duration = time.time() - search_start
        
        context = build_context(results) if results else "(bağlam bulunamadı)"
        logger.info(f"[RAG-DEBUG] Search took {search_duration:.3f}s. Results: {len(results)}. Context length: {len(context)} chars")
        return context
    except Exception as e:
        logger.warning(f"[RAG-ERROR] Context error after {time.time()-start:.3f}s: {e}")
        return "(bağlam hatası)"

# ========================= MAIN ROUTING LOGIC =========================

def build_prompts(query: str, context: str, external_context: str = None):
    system = (
        "Sen bir üniversite ders seçim asistanısın. Verilen BAĞLAM ve EK BİLGİLER'i "
        "kullanarak öğrencinin sorusunu yanıtla. Kaynaklarını belirt. Tüm cevaplarını %100 Türkçe ver. "
        "BAĞLAM yetersizse bunu söyle ve EK BİLGİLER (Öğrenci Profili) ile genel bilgini kullan. "
        "Sadece ders seçimi ve akademik konularda yardımcı ol."
    )
    user = f"Soru: {query}\n\nBAĞLAM:\n{context}"
    if external_context:
        user += f"\n\nEK BİLGİLER (Öğrenci Profili/Kontenjan):\n{external_context}"
    user += "\n\nCevap:"
    return system, user

async def route_and_answer_stream(query: str, external_context: str = None):
    start_time = time.time()
    router = get_router()
    route_start = time.time()
    try:
        route_result = router(query)
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

    context = get_rag_context(query)
    llm_url = config.EASY_LLM_URL if route_name == "easy" else config.HARD_LLM_URL
    system, user = build_prompts(query, context, external_context)
    
    logger.info(f"[LLM-DEBUG] Calling LLM ({route_name}) at {llm_url}. System prompt length: {len(system)}. User prompt length: {len(user)}")

    try:
        token_count = 0
        first_token_time = None
        async for token in call_llm_stream(llm_url, system, user):
            if first_token_time is None:
                first_token_time = time.time()
                logger.info(f"[LLM-DEBUG] First token received in {first_token_time - start_time:.3f}s")
            token_count += 1
            yield json.dumps({"answer": token}) + "\n"
        
        total_duration = time.time() - start_time
        logger.info(f"[LLM-DEBUG] Stream completed. Total tokens: ~{token_count}. Total duration: {total_duration:.3f}s")
    except Exception as e:
        logger.error(f"[LLM-ERROR] Streaming failed after {time.time()-start_time:.3f}s: {e}")
        yield json.dumps({"answer": f"\n[Hata: {str(e)}]"}) + "\n"

async def route_and_answer(query: str, external_context: str = None) -> Dict[str, Any]:
    """Synchronous version for CLI or non-stream use."""
    start = time.time()
    router = get_router()
    route_result = router(query)
    route_name = route_result.name if route_result else None

    if route_name is None:
        return {"query": query, "answer": REJECT_MESSAGE, "route": "rejected", "timing": {"total": time.time()-start}}

    context = get_rag_context(query)
    llm_url = config.EASY_LLM_URL if route_name == "easy" else config.HARD_LLM_URL
    system, user = build_prompts(query, context, external_context)
    
    try:
        answer = await call_llm(llm_url, system, user)
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