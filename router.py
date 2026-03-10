#!/usr/bin/env python3
"""
Semantic Router for RAG — Routes queries to Easy LLM, Hard LLM, or Reject.

Ders seçim domain'ine özelleştirilmiş. Kullanıcı routing'den habersiz,
sistem otomatik olarak soruyu uygun LLM'e yönlendirir.

Architecture:
  User Query → Semantic Router (~5ms) → Easy LLM / Hard LLM / Reject
"""

import asyncio
import time
import logging
from typing import Dict, Any, Tuple

from semantic_router import Route, RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

import aiohttp

from config import config
from tool_calling import answer_with_tools, _get_model_name

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
# Ders seçim domain'ine özel route tanımları.
# Eşleşmeyen sorular otomatik olarak reddedilir.

easy_route = Route(
    name="easy",
    utterances=[
        # Basit bilgi soruları — kısa, doğrudan yanıt
        "Bu dersin kredisi kaç?",
        "Dersin saati ne zaman?",
        "Dersin kontenjanı ne kadar?",
        "Sınav tarihi ne zaman?",
        "Ders saatleri nelerdir?",
        "Ödev teslim tarihi ne?",
        "Devam zorunluluğu var mı?",
        "Dersin hocası kim?",
        "Ders hangi gün?",
        "Dersin ön koşulu ne?",
        "Kontenjan doldu mu?",
        "Hangi dersler seçmeli?",
        "Dersin dili ne?",
        "Bu ders online mı?",
        "Dersin kodu ne?",
        # Ek Türkçe basit sorular
        "Bu ders kaçıncı sınıf dersi?",
        "Vize tarihi ne zaman?",
        "Final sınavı ne zaman?",
        "Dersin AKTS değeri kaç?",
        "Bu ders zorunlu mu?",
        "Ders hangi bölüme ait?",
        "Dersin sınıfı nerede?",
        "Ders hangi fakülteye bağlı?",
        "Bu dersi hangi hoca veriyor?",
        "Bu dersin bütünleme sınavı var mı?",
        "Kayıt yenileme tarihi ne zaman?",
        "Dersin başlangıç tarihi ne?",
        "Bu dersi kimler alabilir?",
        "Proje teslim tarihi ne?",
        "Labın saati ne zaman?",
        "Dersin kitabı ne?",
        "Bu dönem açılan dersler hangileri?",
        "Sınav yeri neresi?",
        "Not ortalaması nasıl hesaplanır?",
        "Transkript nasıl alınır?",
        # English simple questions
        "What is the exam schedule?",
        "When does registration end?",
        "What are the course prerequisites?",
        "How many credits is this course?",
        "What is the grading policy?",
        "What is the attendance policy?",
        "When is the homework deadline?",
        "Who is the instructor for this course?",
        "What time is the lecture?",
        "Where is the classroom?",
        "Is this course mandatory or elective?",
        "What is the course code?",
        "When is the final exam?",
        "When is the midterm exam?",
        "What textbook does this course use?",
        "How is the grade calculated?",
        "Is there a lab session for this course?",
        "What are the office hours?",
        "How do I enroll in this course?",
        "What is the course capacity?",
    ],
)

hard_route = Route(
    name="hard",
    utterances=[
        # Analiz, karşılaştırma, planlama soruları — kapsamlı yanıt
        "Bu dersleri alırsam müfredata uyar mı?",
        "Hangi seçmeli dersler daha uygun olur?",
        "Bu iki dersi aynı dönem alabilir miyim?",
        "Ders programımı nasıl optimize edebilirim?",
        "Bu dersler arasındaki farkları açıklar mısın?",
        "Mezuniyet için hangi dersleri almalıyım?",
        "Ders yükümü nasıl dengeleyebilirim?",
        "Bu bölümün müfredatını analiz et",
        "Seçmeli derslerin avantaj ve dezavantajlarını karşılaştır",
        "Bu ders planıyla kaç dönemde mezun olurum?",
        "Bu derslerin birbirleriyle ilişkisini açıkla",
        # Ek Türkçe analiz/planlama soruları
        "Çift anadal yapmak istiyorum, ders planımı nasıl ayarlamalıyım?",
        "Yandal programına uygun ders seçimi nasıl olmalı?",
        "Bu dönem hangi dersleri almam daha mantıklı?",
        "Staj dönemimde hangi dersleri alabilirim?",
        "Mezuniyetimi uzatmadan hangi dersleri ekleyebilirim?",
        "Bu dersler kariyer hedeflerime uygun mu?",
        "Ders çakışmalarını nasıl çözebilirim?",
        "Tekrar almam gereken derslerle programımı nasıl düzenlemeliyim?",
        "Erasmus döneminde hangi dersleri saydırabilirim?",
        "Üst dönemden ders almak için ne yapmalıyım?",
        "Bu derslerin iş hayatına katkısı ne olur?",
        "Bölüm değiştirirsem hangi derslerim sayılır?",
        "GPA'mı yükseltmek için hangi dersleri seçmeliyim?",
        "Bu ders planıyla yaz okuluna ihtiyacım olur mu?",
        "Zorunlu ve seçmeli dersleri nasıl dengeleyebilirim?",
        "Bu alandaki dersler arasında en faydalısı hangisi?",
        "Lisansüstüne hazırlık için hangi dersleri önerirsin?",
        # English analysis/planning questions
        "Compare the advantages of these elective courses",
        "Analyze my course plan for graduation requirements",
        "Explain the differences between these two courses",
        "Help me plan my semester schedule",
        "What are the trade-offs between these courses?",
        "Which electives would be best for my career goals?",
        "Can I take these two courses in the same semester?",
        "How should I plan my courses for a double major?",
        "What is the best strategy to graduate on time?",
        "How do these courses relate to each other?",
        "Which courses should I prioritize this semester?",
        "Recommend a balanced course load for next semester",
        "What courses can I take during my Erasmus exchange?",
        "How will dropping this course affect my graduation?",
        "What electives complement my major the best?",
        "Should I take summer school or overload next semester?",
        "Analyze the difficulty level of this course combination",
        "How can I improve my GPA with course selection?",
    ],
)


# ========================= ROUTER SETUP =========================

_router_instance = None


def get_router() -> RouteLayer:
    """Get or create the semantic router (singleton)."""
    global _router_instance
    if _router_instance is None:
        logger.info(f"Initializing semantic router with {config.EMBED_MODEL}...")
        encoder = HuggingFaceEncoder(name=config.EMBED_MODEL)
        _router_instance = RouteLayer(
            encoder=encoder,
            routes=[easy_route, hard_route],
        )
        logger.info("Router ready!")
    return _router_instance


# ========================= LLM CLIENT =========================

async def call_llm(base_url: str, system: str, user: str) -> Tuple[str, str]:
    """Send chat request to a llama.cpp server. Returns (answer, model_name)."""
    timeout = aiohttp.ClientTimeout(total=config.LLAMA_CPP_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = await _get_model_name(session, base_url)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": config.LLAMA_CPP_TEMPERATURE,
            "max_tokens": config.LLAMA_CPP_MAX_TOKENS,
            "stream": False,
        }

        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"LLM error: {resp.status} - {error}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"], model


# ========================= RAG INTEGRATION =========================

def get_rag_context(query: str, top_k: int = 5) -> str:
    """Search Qdrant and return context."""
    try:
        from rag_qdrant import VectorIndex, build_context
        index = VectorIndex()
        index.ensure_ready()
        results = index.search(query, top_k=top_k)
        if results:
            return build_context(results)
    except Exception as e:
        logger.warning(f"RAG context error: {e}")
    return "(bağlam bulunamadı)"


# ========================= MAIN ROUTING LOGIC =========================

async def route_and_answer(query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Route a query and get answer from the appropriate LLM.
    Routing is automatic and invisible to the user.
    """
    start = time.time()

    # Step 1: Route the query (~5ms)
    router = get_router()
    route_start = time.time()
    route_result = router(query)
    route_time = time.time() - route_start
    route_name = route_result.name if route_result else None

    if verbose:
        logger.info(f"Route: {route_name or 'REJECTED'} ({route_time*1000:.1f}ms)")

    # Step 2: REJECT if no route matched
    if route_name is None:
        total_time = time.time() - start
        return {
            "query": query,
            "answer": REJECT_MESSAGE,
            "route": "rejected",
            "timing": {
                "route": route_time,
                "rag": 0,
                "llm": 0,
                "total": total_time,
            },
            "debug": {
                "route": "rejected",
                "llm_url": None,
                "llm_model": None,
                "tool_calling": {
                    "triggered": False,
                    "tool_name": None,
                    "course_code": None,
                    "quota_needed": False,
                },
                "rag_chunks": 0,
                "timing": {
                    "route_ms": round(route_time * 1000, 1),
                    "rag_ms": 0,
                    "tool_ms": 0,
                    "llm_ms": 0,
                    "total_ms": round(total_time * 1000, 1),
                },
            },
        }

    # Step 3: Get RAG context from Qdrant
    rag_start = time.time()
    context = get_rag_context(query)
    rag_time = time.time() - rag_start
    # Count RAG chunks (context paragraphs separated by dividers)
    rag_chunks = context.count("---") + 1 if context and context != "(bağlam bulunamadı)" else 0

    if verbose:
        logger.info(f"RAG context: {rag_chunks} chunks ({rag_time:.3f}s)")

    # Step 4: Pick LLM based on route
    if route_name == "easy":
        llm_url = config.EASY_LLM_URL
    else:
        llm_url = config.HARD_LLM_URL

    # Step 5: Tool calling — check if quota data is needed
    tool_start = time.time()
    tool_info = {"tool_used": None, "tool_result": None, "course_code": None, "quota_needed": False}
    try:
        system_prompt = (
            "Sen bir üniversite ders seçim asistanısın. Verilen BAĞLAM'ı kullanarak "
            "öğrencinin sorusunu yanıtla. Kaynaklarını belirt. "
            "BAĞLAM yetersizse bunu söyle ve genel bilgini ekle. "
            "Sadece ders seçimi ve akademik konularda yardımcı ol."
        )
        tool_result = await answer_with_tools(
            query=query,
            context=context,
            llm_url=llm_url,
            system_prompt=system_prompt,
        )
        context = tool_result["enriched_context"]
        tool_info = {
            "tool_used": tool_result["tool_used"],
            "tool_result": tool_result["tool_result"],
            "course_code": tool_result.get("course_code"),
            "quota_needed": tool_result.get("quota_needed", False),
        }
    except Exception as e:
        logger.warning(f"Tool calling failed (non-fatal): {e}")
    tool_time = time.time() - tool_start

    if verbose and tool_info["tool_used"]:
        logger.info(f"Tool '{tool_info['tool_used']}' executed in {tool_time:.3f}s")

    # Step 6: Build final prompt with enriched context
    user_prompt = f"Soru: {query}\n\nBAĞLAM:\n{context}\n\nCevap:"

    # Step 7: Call LLM
    llm_start = time.time()
    llm_model = "unknown"
    try:
        answer, llm_model = await call_llm(llm_url, system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer = f"Üzgünüm, şu anda yanıt üretemiyorum. Lütfen tekrar deneyin."
    llm_time = time.time() - llm_start

    total_time = time.time() - start

    if verbose:
        logger.info(f"LLM [{llm_model}] response in {llm_time:.3f}s | Total: {total_time:.3f}s")

    return {
        "query": query,
        "answer": answer,
        "route": route_name,
        "tool": tool_info,
        "timing": {
            "route": route_time,
            "rag": rag_time,
            "tool": tool_time,
            "llm": llm_time,
            "total": total_time,
        },
        "debug": {
            "route": route_name,
            "llm_url": llm_url,
            "llm_model": llm_model,
            "tool_calling": {
                "triggered": tool_info["tool_used"] is not None,
                "tool_name": tool_info["tool_used"],
                "course_code": tool_info.get("course_code"),
                "quota_needed": tool_info.get("quota_needed", False),
            },
            "rag_chunks": rag_chunks,
            "timing": {
                "route_ms": round(route_time * 1000, 1),
                "rag_ms": round(rag_time * 1000, 1),
                "tool_ms": round(tool_time * 1000, 1),
                "llm_ms": round(llm_time * 1000, 1),
                "total_ms": round(total_time * 1000, 1),
            },
        },
    }


# ========================= CLI =========================

def cmd_ask(args):
    """Ask a question through the router."""
    result = asyncio.run(route_and_answer(args.question))

    print(f"\n{'='*60}")
    print(f"  Query:  {result['query']}")
    print(f"  Route:  {result['route']}")
    print(f"  Time:   {result['timing']['total']:.3f}s")
    print(f"{'='*60}")
    print(f"\n{result['answer']}\n")


def cmd_test(args):
    """Test the router with example queries."""
    router = get_router()

    test_queries = [
        # EASY
        ("Bu dersin kredisi kaç?", "easy"),
        ("Sınav tarihi ne zaman?", "easy"),
        ("Dersin hocası kim?", "easy"),
        # HARD
        ("Hangi seçmeli dersler daha uygun olur?", "hard"),
        ("Ders programımı nasıl optimize edebilirim?", "hard"),
        ("Bu derslerin farkları neler?", "hard"),
        # REJECTED
        ("Bugün hava nasıl?", "rejected"),
        ("Bana bir fıkra anlat", "rejected"),
        ("Futbol maçı kaç kaç bitti?", "rejected"),
        ("Bitcoin fiyatı ne kadar?", "rejected"),
    ]

    print(f"\n{'='*70}")
    print(f"  {'QUERY':<45} {'EXPECTED':<10} {'GOT':<10} {'OK'}")
    print(f"{'='*70}")

    correct = 0
    for query, expected in test_queries:
        result = router(query)
        got = result.name if result else "rejected"
        ok = "✅" if got == expected else "❌"
        if got == expected:
            correct += 1
        print(f"  {query:<45} {expected:<10} {got:<10} {ok}")

    print(f"{'='*70}")
    print(f"  Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.0f}%)")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ders Seçim Chatbot — Semantic Router")
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str)
    ask_parser.set_defaults(func=cmd_ask)

    test_parser = subparsers.add_parser("test", help="Test routing accuracy")
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()