#!/usr/bin/env python3
"""
Tool Calling Module — Lesson Quota Check

Provides tool calling functionality for the RAG chatbot.
The LLM can decide to check course quota/capacity data from a
MariaDB-backed API endpoint.

Two-pass approach (works with any LLM):
  1. First pass: Extract course code from user query
  2. If found: Call quota API to get real-time data
  3. Second pass: Answer with RAG context + quota data

Usage:
    from tool_calling import answer_with_tools
    result = await answer_with_tools(query, context, llm_url)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp

from config import config

logger = logging.getLogger(__name__)


# ========================= QUOTA TOOL =========================

EXTRACT_COURSE_CODE_PROMPT = (
    "Sen bir ders kodu çıkarıcısın. Kullanıcının sorusundan ders kodunu çıkar.\n"
    "Ders kodu genellikle harf ve rakamlardan oluşur (örn: BIL101, MAT201, FIZ102).\n"
    "Eğer soruda bir ders kodu varsa sadece ders kodunu yaz.\n"
    "Eğer soruda ders kodu yoksa sadece 'YOK' yaz.\n"
    "Başka hiçbir şey yazma, açıklama yapma."
)

NEEDS_QUOTA_CHECK_PROMPT = (
    "Kullanıcı sorusu bir dersin kontenjanı, kapasitesi, kaç kişinin aldığı, "
    "dolup dolmadığı veya boş yer olup olmadığı hakkında mı?\n"
    "Sadece 'EVET' veya 'HAYIR' yaz. Başka bir şey yazma."
)


async def _call_llm_simple(base_url: str, system: str, user: str) -> str:
    """Simple LLM call for tool orchestration (short responses)."""
    timeout = aiohttp.ClientTimeout(total=config.QUOTA_API_TIMEOUT + 20)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Auto-detect model
        model = "default"
        try:
            async with session.get(f"{base_url}/v1/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data") and len(data["data"]) > 0:
                        model = data["data"][0]["id"]
        except Exception:
            pass

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,  # Deterministic for extraction
            "max_tokens": 50,    # Short responses only
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
            return data["choices"][0]["message"]["content"].strip()


async def needs_quota_check(query: str, llm_url: str) -> bool:
    """
    Ask the LLM if the query is about course quota/capacity.
    Returns True if quota data should be fetched.
    """
    try:
        response = await _call_llm_simple(
            llm_url, NEEDS_QUOTA_CHECK_PROMPT, query
        )
        result = response.strip().upper()
        logger.info(f"Quota check needed? LLM says: {result}")
        return "EVET" in result
    except Exception as e:
        logger.warning(f"Quota check detection failed: {e}")
        return False


async def extract_course_code(query: str, llm_url: str) -> Optional[str]:
    """
    Extract course code from user query using LLM.
    Returns the course code string or None if not found.
    """
    try:
        response = await _call_llm_simple(
            llm_url, EXTRACT_COURSE_CODE_PROMPT, query
        )
        code = response.strip().upper()

        if "YOK" in code or len(code) < 3 or len(code) > 15:
            logger.info(f"No course code found in query: '{query}'")
            return None

        # Clean up: remove any extra text, keep only alphanumeric
        code = re.sub(r'[^A-Z0-9ÇĞİÖŞÜ ]', '', code).strip()

        if code:
            logger.info(f"Extracted course code: {code}")
            return code

        return None

    except Exception as e:
        logger.warning(f"Course code extraction failed: {e}")
        return None


async def check_quota(course_code: str) -> Dict[str, Any]:
    """
    Fetch quota data for a course from the MariaDB-backed API.

    Args:
        course_code: Course code (e.g., "BIL101")

    Returns:
        Dict with quota info: {
            "course_code": str,
            "capacity": int,     # Maximum students
            "enrolled": int,     # Currently enrolled
            "available": int,    # Remaining spots
            "is_full": bool,     # Whether quota is full
            "raw_data": dict,    # Full API response
            "error": str | None  # Error message if failed
        }
    """
    url = f"{config.QUOTA_API_URL}/{course_code}"
    timeout = aiohttp.ClientTimeout(total=config.QUOTA_API_TIMEOUT)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(
                        f"Quota API error for {course_code}: "
                        f"{resp.status} - {error_text}"
                    )
                    return {
                        "course_code": course_code,
                        "error": f"API error: {resp.status}",
                        "raw_data": None,
                    }

                data = await resp.json()

                # --------------------------------------------------
                # ADAPT THESE FIELD NAMES to match your actual API
                # response structure from MariaDB.
                # --------------------------------------------------
                capacity = data.get("capacity", data.get("kontenjan", 0))
                enrolled = data.get("enrolled", data.get("kayitli", 0))
                available = capacity - enrolled

                result = {
                    "course_code": course_code,
                    "capacity": capacity,
                    "enrolled": enrolled,
                    "available": max(0, available),
                    "is_full": available <= 0,
                    "raw_data": data,
                    "error": None,
                }
                logger.info(
                    f"Quota for {course_code}: "
                    f"{enrolled}/{capacity} (available: {available})"
                )
                return result

    except aiohttp.ClientError as e:
        logger.error(f"Quota API connection error for {course_code}: {e}")
        return {
            "course_code": course_code,
            "error": f"Connection error: {str(e)}",
            "raw_data": None,
        }
    except Exception as e:
        logger.error(f"Quota check failed for {course_code}: {e}")
        return {
            "course_code": course_code,
            "error": str(e),
            "raw_data": None,
        }


def format_quota_for_prompt(quota_data: Dict[str, Any]) -> str:
    """Format quota data as context text for the LLM prompt."""
    if quota_data.get("error"):
        return (
            f"\n\n[KONTENJAN BİLGİSİ — {quota_data['course_code']}]\n"
            f"Kontenjan bilgisi alınamadı: {quota_data['error']}\n"
        )

    status = "DOLU ❌" if quota_data["is_full"] else "BOŞ YER VAR ✅"
    return (
        f"\n\n[KONTENJAN BİLGİSİ — {quota_data['course_code']}]\n"
        f"Kontenjan: {quota_data['capacity']}\n"
        f"Kayıtlı Öğrenci: {quota_data['enrolled']}\n"
        f"Boş Yer: {quota_data['available']}\n"
        f"Durum: {status}\n"
    )


# ========================= TOOL CALLING ORCHESTRATOR =========================

async def answer_with_tools(
    query: str,
    context: str,
    llm_url: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """
    Orchestrate tool calling: check if quota is needed, fetch it,
    then build final prompt with all data.

    Returns:
        Dict with:
            - "enriched_context": context + quota data (if any)
            - "tool_used": name of tool used or None
            - "tool_result": raw tool result or None
    """
    tool_result = None
    tool_used = None
    enriched_context = context

    # Step 1: Does the query need quota information?
    quota_needed = await needs_quota_check(query, llm_url)

    if quota_needed:
        # Step 2: Extract course code
        course_code = await extract_course_code(query, llm_url)

        if course_code:
            # Step 3: Fetch quota data
            tool_used = "check_quota"
            tool_result = await check_quota(course_code)

            # Step 4: Add quota data to context
            quota_text = format_quota_for_prompt(tool_result)
            enriched_context = context + quota_text

            logger.info(
                f"Tool '{tool_used}' called for {course_code}: "
                f"{json.dumps(tool_result, ensure_ascii=False, default=str)}"
            )
        else:
            logger.info("Quota check needed but no course code found in query")

    return {
        "enriched_context": enriched_context,
        "tool_used": tool_used,
        "tool_result": tool_result,
    }
