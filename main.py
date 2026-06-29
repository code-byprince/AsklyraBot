#!/usr/bin/env python3
"""
==============================================================================
  Telegram AI Chatbot — Production-Ready Single-File Implementation
  Author : Senior Python AI Engineer
  Python : 3.12+
  Libs   : python-telegram-bot v22+, aiohttp, aiosqlite, openai-compatible
  Deploy : Render (set BOT_TOKEN + OPENROUTER_API_KEY env vars)
==============================================================================
"""

# ─────────────────────────────── stdlib ──────────────────────────────────────
import asyncio
import io
import json
import logging
import math
import mimetypes
import os
import re
import sqlite3
import sys
import tempfile
import textwrap
import time
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any

# ─────────────────────────────── third-party ─────────────────────────────────
import aiohttp
import aiosqlite
import docx                          # python-docx
import pypdf                         # pypdf
from telegram import (
    BotCommand,
    Chat,
    Document,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
    Voice,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.helpers import escape_markdown

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (all values read from environment — never hardcoded)
# ══════════════════════════════════════════════════════════════════════════════

BOT_TOKEN: str = os.environ["BOT_TOKEN"]                     # Telegram token
OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]   # OpenRouter key

# Optional env vars with sane defaults
ADMIN_IDS: set[int] = {
    int(x) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip()
}
DB_PATH: str = os.environ.get("DB_PATH", "chatbot.db")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
MAX_HISTORY_TOKENS: int = int(os.environ.get("MAX_HISTORY_TOKENS", "8000"))
SUMMARY_THRESHOLD: int = int(os.environ.get("SUMMARY_THRESHOLD", "20"))
RATE_LIMIT_MESSAGES: int = int(os.environ.get("RATE_LIMIT_MESSAGES", "20"))
RATE_LIMIT_WINDOW: int = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))   # seconds
DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "openrouter/auto")
OPENROUTER_BASE: str = "https://openrouter.ai/api/v1"
MAX_MESSAGE_LENGTH: int = 4000          # Telegram limit is 4096; keep buffer
CLEANUP_DAYS: int = int(os.environ.get("CLEANUP_DAYS", "30"))

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("tgbot")

# ══════════════════════════════════════════════════════════════════════════════
#  FREE-MODEL REGISTRY  (auto-updated at runtime via OpenRouter /models)
# ══════════════════════════════════════════════════════════════════════════════

# Static fallback list in case the API call to /models fails
STATIC_FREE_MODELS: list[str] = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b:free",
]

# Runtime state — populated once on startup
_free_models: list[str] = list(STATIC_FREE_MODELS)
_model_fetch_ts: float = 0.0
_MODEL_TTL: float = 3600.0    # refresh free-model list every hour

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE  (SQLite via aiosqlite for async I/O)
# ══════════════════════════════════════════════════════════════════════════════

CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY,
    username      TEXT,
    first_name    TEXT,
    last_name     TEXT,
    language_code TEXT,
    is_admin      INTEGER DEFAULT 0,
    is_banned     INTEGER DEFAULT 0,
    preferred_model TEXT,
    created_at    TEXT DEFAULT (datetime('now')),
    last_seen     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER NOT NULL,
    user_id       INTEGER NOT NULL,
    role          TEXT NOT NULL,          -- 'user' | 'assistant' | 'system'
    content       TEXT NOT NULL,
    tokens        INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER NOT NULL,
    user_id       INTEGER NOT NULL,
    summary       TEXT NOT NULL,
    covered_up_to INTEGER NOT NULL,       -- last conversation.id covered
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rate_limits (
    user_id       INTEGER PRIMARY KEY,
    message_count INTEGER DEFAULT 0,
    window_start  REAL    DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stats (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    chat_id        INTEGER NOT NULL,
    model_used     TEXT,
    prompt_tokens  INTEGER DEFAULT 0,
    reply_tokens   INTEGER DEFAULT 0,
    latency_ms     INTEGER DEFAULT 0,
    created_at     TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_conversations_chat
    ON conversations (chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_stats_user
    ON stats (user_id);
"""


async def init_db() -> None:
    """Create all tables and indexes if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_TABLES_SQL)
        await db.commit()
    logger.info("Database initialised at %s", DB_PATH)


async def upsert_user(update: Update) -> None:
    """Insert or update a user record from Telegram update metadata."""
    user = update.effective_user
    if user is None:
        return
    is_admin = 1 if user.id in ADMIN_IDS else 0
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO users (user_id, username, first_name, last_name,
                               language_code, is_admin, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                username      = excluded.username,
                first_name    = excluded.first_name,
                last_name     = excluded.last_name,
                language_code = excluded.language_code,
                is_admin      = excluded.is_admin,
                last_seen     = excluded.last_seen
            """,
            (
                user.id,
                user.username,
                user.first_name,
                user.last_name,
                user.language_code,
                is_admin,
            ),
        )
        await db.commit()


async def get_user(user_id: int) -> dict | None:
    """Return a user row as a dict, or None if not found."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def set_preferred_model(user_id: int, model: str) -> None:
    """Persist the user's chosen model."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET preferred_model = ? WHERE user_id = ?",
            (model, user_id),
        )
        await db.commit()


async def is_banned(user_id: int) -> bool:
    """Return True if the user is banned."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT is_banned FROM users WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            return bool(row[0]) if row else False


async def ban_user(user_id: int, banned: bool) -> None:
    """Set or unset a ban on a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET is_banned = ? WHERE user_id = ?",
            (1 if banned else 0, user_id),
        )
        await db.commit()


# ── Conversation history ──────────────────────────────────────────────────────

async def add_message(
    chat_id: int, user_id: int, role: str, content: str, tokens: int = 0
) -> int:
    """Append one message to the conversation log. Returns new row id."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO conversations (chat_id, user_id, role, content, tokens)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_id, user_id, role, content, tokens),
        )
        await db.commit()
        return cur.lastrowid  # type: ignore[return-value]


async def get_history(
    chat_id: int, limit: int = 100
) -> list[dict[str, Any]]:
    """Return recent conversation rows for a chat (oldest first)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, role, content, tokens, created_at
            FROM conversations
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in reversed(rows)]


async def clear_history(chat_id: int, user_id: int) -> int:
    """Delete all messages for a chat. Returns number of deleted rows."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM conversations WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id),
        )
        await db.execute(
            "DELETE FROM summaries WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id),
        )
        await db.commit()
        return cur.rowcount  # type: ignore[return-value]


async def search_history(
    user_id: int, query: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Full-text search over a user's conversation history."""
    pattern = f"%{query}%"
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT chat_id, role, content, created_at
            FROM conversations
            WHERE user_id = ? AND content LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, pattern, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def cleanup_old_conversations(days: int = CLEANUP_DAYS) -> int:
    """Remove conversation rows older than `days` days. Returns deleted count."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM conversations WHERE created_at < ?", (cutoff,)
        )
        await db.commit()
        return cur.rowcount  # type: ignore[return-value]


# ── Summaries ─────────────────────────────────────────────────────────────────

async def get_latest_summary(chat_id: int, user_id: int) -> dict | None:
    """Return the most recent summary row, or None."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM summaries
            WHERE chat_id = ? AND user_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (chat_id, user_id),
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def save_summary(
    chat_id: int, user_id: int, summary: str, covered_up_to: int
) -> None:
    """Persist a new summary record."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO summaries (chat_id, user_id, summary, covered_up_to)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, user_id, summary, covered_up_to),
        )
        await db.commit()


# ── Rate limiting ─────────────────────────────────────────────────────────────

async def check_rate_limit(user_id: int) -> bool:
    """
    Returns True if the user is within the rate limit window.
    Increments counter; resets window when expired.
    """
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT message_count, window_start FROM rate_limits WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            await db.execute(
                "INSERT INTO rate_limits (user_id, message_count, window_start) VALUES (?, 1, ?)",
                (user_id, now),
            )
            await db.commit()
            return True

        count, window_start = row
        if now - window_start > RATE_LIMIT_WINDOW:
            # New window
            await db.execute(
                "UPDATE rate_limits SET message_count=1, window_start=? WHERE user_id=?",
                (now, user_id),
            )
            await db.commit()
            return True

        if count >= RATE_LIMIT_MESSAGES:
            return False

        await db.execute(
            "UPDATE rate_limits SET message_count=message_count+1 WHERE user_id=?",
            (user_id,),
        )
        await db.commit()
        return True


# ── Stats ─────────────────────────────────────────────────────────────────────

async def record_stat(
    user_id: int,
    chat_id: int,
    model: str,
    prompt_tokens: int,
    reply_tokens: int,
    latency_ms: int,
) -> None:
    """Write one inference stat row."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO stats
              (user_id, chat_id, model_used, prompt_tokens, reply_tokens, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, chat_id, model, prompt_tokens, reply_tokens, latency_ms),
        )
        await db.commit()


async def get_stats(user_id: int) -> dict[str, Any]:
    """Aggregate statistics for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT
                COUNT(*)           AS total_requests,
                SUM(prompt_tokens) AS total_prompt_tokens,
                SUM(reply_tokens)  AS total_reply_tokens,
                AVG(latency_ms)    AS avg_latency_ms,
                MIN(created_at)    AS first_request,
                MAX(created_at)    AS last_request
            FROM stats WHERE user_id = ?
            """,
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
        async with db.execute(
            """
            SELECT model_used, COUNT(*) AS cnt
            FROM stats WHERE user_id = ?
            GROUP BY model_used ORDER BY cnt DESC LIMIT 3
            """,
            (user_id,),
        ) as cur:
            top_models = await cur.fetchall()
    return {
        **(dict(row) if row else {}),
        "top_models": [dict(r) for r in top_models],
    }


async def get_global_stats() -> dict[str, Any]:
    """Admin-only aggregate stats."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT COUNT(*) AS total_users FROM users"
        ) as cur:
            users_row = await cur.fetchone()
        async with db.execute(
            "SELECT COUNT(*) AS total_messages FROM conversations"
        ) as cur:
            msgs_row = await cur.fetchone()
        async with db.execute(
            "SELECT COUNT(*) AS total_requests, SUM(reply_tokens) AS total_tokens FROM stats"
        ) as cur:
            stats_row = await cur.fetchone()
    return {
        "total_users": dict(users_row)["total_users"] if users_row else 0,
        "total_messages": dict(msgs_row)["total_messages"] if msgs_row else 0,
        "total_requests": dict(stats_row)["total_requests"] if stats_row else 0,
        "total_tokens": dict(stats_row)["total_tokens"] if stats_row else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  OPENROUTER  —  free-model discovery + chat completion
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_free_models(session: aiohttp.ClientSession) -> list[str]:
    """
    Query OpenRouter /models, filter models whose ID ends with ':free',
    and return them sorted alphabetically.  Falls back to STATIC_FREE_MODELS.
    """
    global _free_models, _model_fetch_ts
    try:
        async with session.get(
            f"{OPENROUTER_BASE}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                ids = sorted(
                    m["id"]
                    for m in data.get("data", [])
                    if m.get("id", "").endswith(":free")
                )
                if ids:
                    _free_models = ids
                    _model_fetch_ts = time.time()
                    logger.info("Fetched %d free models from OpenRouter", len(ids))
                    return ids
    except Exception as exc:
        logger.warning("Could not fetch free models: %s", exc)
    return _free_models


async def get_free_models(session: aiohttp.ClientSession) -> list[str]:
    """Return cached free models, refreshing if TTL expired."""
    if time.time() - _model_fetch_ts > _MODEL_TTL:
        await fetch_free_models(session)
    return _free_models


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tgbot",
        "X-Title": "Telegram AI Chatbot",
    }


async def chat_completion(
    session: aiohttp.ClientSession,
    messages: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    stream: bool = False,
) -> tuple[str, str, int, int]:
    """
    Call the OpenRouter chat-completion endpoint with automatic fallback.

    Returns:
        (reply_text, model_used, prompt_tokens, completion_tokens)

    Raises:
        RuntimeError if all models fail.
    """
    free_models = await get_free_models(session)

    # Build ordered candidate list
    if model and model not in (DEFAULT_MODEL, "openrouter/auto"):
        candidates = [model] + [m for m in free_models if m != model]
    else:
        candidates = list(free_models)

    # Always append a reliable fallback that OpenRouter routes automatically
    if "openrouter/auto" not in candidates:
        candidates.append("openrouter/auto")

    last_error: Exception | None = None
    for attempt, candidate in enumerate(candidates, 1):
        try:
            payload: dict[str, Any] = {
                "model": candidate,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            if stream:
                return await _stream_completion(session, payload, candidate)
            else:
                return await _standard_completion(session, payload, candidate)
        except Exception as exc:
            logger.warning(
                "Model %s failed (attempt %d): %s", candidate, attempt, exc
            )
            last_error = exc
            await asyncio.sleep(0.5)   # brief pause before next candidate

    raise RuntimeError(
        f"All {len(candidates)} model candidates failed. Last error: {last_error}"
    )


async def _standard_completion(
    session: aiohttp.ClientSession,
    payload: dict[str, Any],
    model: str,
) -> tuple[str, str, int, int]:
    """Non-streaming completion call."""
    async with session.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers=_auth_headers(),
        json=payload,
        timeout=aiohttp.ClientTimeout(total=90),
    ) as resp:
        body = await resp.json()
        if resp.status != 200:
            raise RuntimeError(
                f"HTTP {resp.status}: {body.get('error', {}).get('message', body)}"
            )
        choice = body["choices"][0]
        text: str = choice["message"]["content"] or ""
        usage = body.get("usage", {})
        return (
            text,
            body.get("model", model),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )


async def _stream_completion(
    session: aiohttp.ClientSession,
    payload: dict[str, Any],
    model: str,
) -> tuple[str, str, int, int]:
    """
    Streaming completion — collects all SSE chunks and returns the full text.
    (Telegram bots send 'typing' while waiting; real token-by-token updates
    require editing intermediate messages which is handled in the handler layer.)
    """
    chunks: list[str] = []
    async with session.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers=_auth_headers(),
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            err = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {err[:200]}")
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj["choices"][0].get("delta", {})
                content = delta.get("content") or ""
                chunks.append(content)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    text = "".join(chunks)
    # Token counts unavailable in streaming without an extra call; approximate
    prompt_tokens = sum(len(m.get("content", "").split()) for m in payload["messages"])
    completion_tokens = len(text.split())
    return text, model, prompt_tokens, completion_tokens


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _rough_token_count(text: str) -> int:
    """Rough approximation: 1 token ≈ 4 chars."""
    return math.ceil(len(text) / 4)


async def build_messages(
    chat_id: int,
    user_id: int,
    new_user_content: Any,          # str or list (for multimodal)
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Assemble the messages array to send to the model:
      1. System prompt (with optional summary prefix)
      2. Recent conversation history (trimmed to MAX_HISTORY_TOKENS)
      3. The new user message
    """
    system_parts: list[str] = []

    # Include latest summary if available
    summary_row = await get_latest_summary(chat_id, user_id)
    if summary_row:
        system_parts.append(
            f"[Conversation summary so far]\n{summary_row['summary']}"
        )

    # Base system prompt
    base_system = system_prompt or (
        "You are a helpful, knowledgeable, and friendly AI assistant. "
        "You support multiple languages — always reply in the same language "
        "the user wrote in. Format your replies clearly using Markdown when "
        "it adds value. Be concise yet thorough."
    )
    system_parts.append(base_system)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]

    # Load and trim history
    history = await get_history(chat_id, limit=200)
    token_budget = MAX_HISTORY_TOKENS
    trimmed: list[dict[str, Any]] = []
    for row in reversed(history):   # newest first
        cost = _rough_token_count(str(row["content"]))
        if cost > token_budget:
            break
        token_budget -= cost
        trimmed.append({"role": row["role"], "content": row["content"]})
    messages.extend(reversed(trimmed))  # restore oldest-first order

    # New user turn
    messages.append({"role": "user", "content": new_user_content})
    return messages


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARISATION  (triggered when history > SUMMARY_THRESHOLD messages)
# ══════════════════════════════════════════════════════════════════════════════

async def maybe_summarise(
    session: aiohttp.ClientSession,
    chat_id: int,
    user_id: int,
) -> None:
    """
    If the number of un-summarised messages exceeds SUMMARY_THRESHOLD,
    ask the model to summarise them and persist the result.
    """
    history = await get_history(chat_id, limit=500)
    summary_row = await get_latest_summary(chat_id, user_id)
    covered_up_to = summary_row["covered_up_to"] if summary_row else 0

    unsummarised = [r for r in history if r["id"] > covered_up_to]
    if len(unsummarised) < SUMMARY_THRESHOLD:
        return

    logger.info(
        "Summarising %d messages for chat %d", len(unsummarised), chat_id
    )
    transcript = "\n".join(
        f"{r['role'].upper()}: {r['content']}" for r in unsummarised
    )
    prompt = (
        "Summarise the following conversation concisely, preserving all key "
        "facts, decisions, and context. Use bullet points.\n\n" + transcript
    )
    try:
        summary_text, _, _, _ = await chat_completion(
            session,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        last_id = unsummarised[-1]["id"]
        await save_summary(chat_id, user_id, summary_text, last_id)
        logger.info("Summary saved, covered up to id=%d", last_id)
    except Exception as exc:
        logger.error("Summarisation failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
#  FILE PROCESSING  (PDF, DOCX, TXT, images, voice)
# ══════════════════════════════════════════════════════════════════════════════

async def download_file(bot, file_id: str) -> bytes:
    """Download a Telegram file and return raw bytes."""
    tg_file = await bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(buf)
    return buf.getvalue()


def extract_pdf_text(data: bytes) -> str:
    """Extract plain text from PDF bytes using pypdf."""
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.strip())
    full = "\n\n".join(p for p in pages if p)
    # Truncate to avoid overflowing the context window
    return full[:12000] if len(full) > 12000 else full


def extract_docx_text(data: bytes) -> str:
    """Extract plain text from DOCX bytes using python-docx."""
    doc = docx.Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full = "\n".join(paragraphs)
    return full[:12000] if len(full) > 12000 else full


def extract_txt_text(data: bytes) -> str:
    """Decode a plain-text file, trying common encodings."""
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = data.decode(enc)
            return text[:12000] if len(text) > 12000 else text
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")[:12000]


async def transcribe_voice(
    session: aiohttp.ClientSession,
    audio_bytes: bytes,
    mime: str = "audio/ogg",
) -> str:
    """
    Transcribe voice using OpenRouter's Whisper-compatible endpoint.
    Falls back to a descriptive placeholder if unavailable.
    """
    # OpenRouter wraps OpenAI-compatible audio endpoint
    url = f"{OPENROUTER_BASE}/audio/transcriptions"
    form = aiohttp.FormData()
    form.add_field("model", "openai/whisper-large-v3")
    form.add_field(
        "file",
        audio_bytes,
        filename="voice.ogg",
        content_type=mime,
    )
    try:
        async with session.post(
            url,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            data=form,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("text", "")
    except Exception as exc:
        logger.warning("Voice transcription failed: %s", exc)
    return "[Voice message — transcription unavailable]"


def image_bytes_to_base64_url(data: bytes, mime: str) -> str:
    """Convert image bytes to a data-URL suitable for multimodal messages."""
    import base64
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


# ══════════════════════════════════════════════════════════════════════════════
#  REPLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message at paragraph boundaries, respecting `limit`."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        # Try to split on a newline
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        parts.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return parts


async def safe_reply(
    message: Message,
    text: str,
    parse_mode: str | None = ParseMode.MARKDOWN,
    **kwargs: Any,
) -> None:
    """
    Send a reply, auto-splitting long messages.
    Falls back to plain text on Markdown parse errors.
    """
    chunks = _split_message(text)
    for i, chunk in enumerate(chunks):
        try:
            await message.reply_text(
                chunk,
                parse_mode=parse_mode,
                **kwargs,
            )
        except Exception:
            # Retry as plain text
            try:
                await message.reply_text(chunk, parse_mode=None)
            except Exception as exc2:
                logger.error("Failed to send reply chunk %d: %s", i, exc2)
        if i < len(chunks) - 1:
            await asyncio.sleep(0.3)


# ══════════════════════════════════════════════════════════════════════════════
#  MIDDLEWARE  (ban check + rate limit, called from every public handler)
# ══════════════════════════════════════════════════════════════════════════════

async def guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Return True if the request should proceed.
    Sends an error reply and returns False otherwise.
    """
    user = update.effective_user
    if user is None:
        return False
    await upsert_user(update)
    if await is_banned(user.id):
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "🚫 You have been banned from using this bot."
        )
        return False
    if not await check_rate_limit(user.id):
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"⏳ Slow down! You can send at most {RATE_LIMIT_MESSAGES} messages "
            f"per {RATE_LIMIT_WINDOW} seconds."
        )
        return False
    return True


def is_admin(user_id: int) -> bool:
    """Check in-memory admin set (admins are set via ADMIN_IDS env var)."""
    return user_id in ADMIN_IDS


# ══════════════════════════════════════════════════════════════════════════════
#  COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start — welcome message."""
    if not await guard(update, context):
        return
    user = update.effective_user
    name = user.first_name if user else "there"
    text = (
        f"👋 Hello, *{escape_markdown(name, version=2)}*\\!\n\n"
        "I'm your AI assistant powered by the best free models on OpenRouter\\.\n\n"
        "*Quick commands:*\n"
        "/help — show all commands\n"
        "/new — start a new conversation\n"
        "/models — list available free models\n"
        "/model — change your AI model\n\n"
        "Just send me a message to get started\\! 🚀"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)  # type: ignore[union-attr]


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help — command reference."""
    if not await guard(update, context):
        return
    text = (
        "📖 *Command Reference*\n\n"
        "*/start* — Welcome message\n"
        "*/help* — This help text\n"
        "*/new* — Start a fresh conversation (keeps history)\n"
        "*/clear* — Erase ALL conversation history\n"
        "*/history* — Show recent messages\n"
        "*/stats* — Your usage statistics\n"
        "*/ping* — Check bot latency\n"
        "*/model* — Change AI model\n"
        "*/models* — List all free models\n"
        "*/export* — Export your conversation as a text file\n\n"
        "📎 *Supported file types:* PDF, DOCX, TXT, images, voice\n\n"
        "🔍 *Search history:* /history `keyword`"
    )
    await safe_reply(update.message, text)  # type: ignore[arg-type]


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/new — inject a system separator without erasing history."""
    if not await guard(update, context):
        return
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    user_id = update.effective_user.id  # type: ignore[union-attr]
    await add_message(chat_id, user_id, "system", "--- New conversation started ---")
    await update.message.reply_text("✅ New conversation started! Your history is preserved but the AI will treat this as a fresh session.")  # type: ignore[union-attr]


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/clear — delete all messages (with confirmation)."""
    if not await guard(update, context):
        return
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Yes, clear everything", callback_data="clear_confirm"),
                InlineKeyboardButton("❌ Cancel", callback_data="clear_cancel"),
            ]
        ]
    )
    await update.message.reply_text(  # type: ignore[union-attr]
        "⚠️ Are you sure you want to delete *all* your conversation history?",
        reply_markup=keyboard,
        parse_mode=ParseMode.MARKDOWN,
    )


async def callback_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle clear confirmation inline keyboard."""
    query = update.callback_query
    await query.answer()
    if query.data == "clear_confirm":
        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        user_id = update.effective_user.id  # type: ignore[union-attr]
        deleted = await clear_history(chat_id, user_id)
        await query.edit_message_text(
            f"🗑️ Cleared {deleted} messages. Starting fresh!"
        )
    else:
        await query.edit_message_text("Cancelled. Your history is safe. ✅")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/history [keyword] — show recent messages or search."""
    if not await guard(update, context):
        return
    user_id = update.effective_user.id  # type: ignore[union-attr]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]

    query_terms = " ".join(context.args or []).strip()
    if query_terms:
        # Search mode
        results = await search_history(user_id, query_terms, limit=8)
        if not results:
            await update.message.reply_text(f"🔍 No results for *{query_terms}*.", parse_mode=ParseMode.MARKDOWN)  # type: ignore[union-attr]
            return
        lines = [f"🔍 *Search results for* `{query_terms}`:\n"]
        for r in results:
            ts = r["created_at"][:16]
            preview = r["content"][:120].replace("\n", " ")
            lines.append(f"[{ts}] *{r['role']}*: {preview}")
        await safe_reply(update.message, "\n".join(lines))  # type: ignore[arg-type]
        return

    # Recent history mode
    rows = await get_history(chat_id, limit=10)
    if not rows:
        await update.message.reply_text("No conversation history yet.")  # type: ignore[union-attr]
        return
    lines = ["📜 *Last 10 messages:*\n"]
    for r in rows:
        ts = r["created_at"][:16]
        preview = r["content"][:100].replace("\n", " ")
        lines.append(f"[{ts}] *{r['role']}*: {preview}")
    await safe_reply(update.message, "\n".join(lines))  # type: ignore[arg-type]


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/stats — show personal (and global for admins) statistics."""
    if not await guard(update, context):
        return
    user_id = update.effective_user.id  # type: ignore[union-attr]
    s = await get_stats(user_id)

    lines = [
        "📊 *Your Statistics*\n",
        f"Total requests: `{s.get('total_requests', 0)}`",
        f"Prompt tokens: `{s.get('total_prompt_tokens', 0)}`",
        f"Reply tokens: `{s.get('total_reply_tokens', 0)}`",
        f"Avg latency: `{int(s.get('avg_latency_ms') or 0)} ms`",
        f"First request: `{s.get('first_request', 'N/A')}`",
        f"Last request: `{s.get('last_request', 'N/A')}`",
    ]
    top = s.get("top_models", [])
    if top:
        lines.append("\n🏆 *Top models used:*")
        for m in top:
            lines.append(f"  • `{m['model_used']}` — {m['cnt']} calls")

    if is_admin(user_id):
        gs = await get_global_stats()
        lines += [
            "\n\n🌐 *Global Stats (Admin)*",
            f"Users: `{gs['total_users']}`",
            f"Messages: `{gs['total_messages']}`",
            f"API calls: `{gs['total_requests']}`",
            f"Tokens generated: `{gs['total_tokens']}`",
        ]

    await safe_reply(update.message, "\n".join(lines))  # type: ignore[arg-type]


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/ping — measure round-trip latency."""
    if not await guard(update, context):
        return
    t0 = time.perf_counter()
    msg = await update.message.reply_text("🏓 Pong...")  # type: ignore[union-attr]
    ms = int((time.perf_counter() - t0) * 1000)
    await msg.edit_text(f"🏓 Pong! `{ms} ms`", parse_mode=ParseMode.MARKDOWN)


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/model [model_id] — view or change the preferred model."""
    if not await guard(update, context):
        return
    user_id = update.effective_user.id  # type: ignore[union-attr]
    user_row = await get_user(user_id)
    current = (user_row or {}).get("preferred_model") or DEFAULT_MODEL

    args = context.args or []
    if not args:
        # Show current model
        await update.message.reply_text(  # type: ignore[union-attr]
            f"🤖 Current model: `{current}`\n\nUse `/model <model_id>` to change, "
            "or `/models` to list free models.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    new_model = args[0].strip()
    await set_preferred_model(user_id, new_model)
    await update.message.reply_text(  # type: ignore[union-attr]
        f"✅ Model changed to `{new_model}`", parse_mode=ParseMode.MARKDOWN
    )


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/models — list all available free models from OpenRouter."""
    if not await guard(update, context):
        return
    session: aiohttp.ClientSession = context.bot_data["session"]
    models = await get_free_models(session)
    if not models:
        await update.message.reply_text("Could not retrieve model list.")  # type: ignore[union-attr]
        return
    lines = ["🆓 *Available Free Models:*\n"]
    for m in models:
        lines.append(f"  • `{m}`")
    lines.append(
        f"\nUse `/model <model_id>` to switch\\.\n_Last updated: {datetime.now():%H:%M:%S}_"
    )
    await safe_reply(update.message, "\n".join(lines))  # type: ignore[arg-type]


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/export — send conversation history as a .txt file download."""
    if not await guard(update, context):
        return
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    user_id = update.effective_user.id  # type: ignore[union-attr]
    rows = await get_history(chat_id, limit=1000)
    if not rows:
        await update.message.reply_text("No history to export.")  # type: ignore[union-attr]
        return
    lines: list[str] = [
        f"Conversation Export — {datetime.now():%Y-%m-%d %H:%M:%S}\n",
        "=" * 60 + "\n",
    ]
    for r in rows:
        lines.append(f"[{r['created_at']}] {r['role'].upper()}:\n{r['content']}\n\n")
    content = "".join(lines).encode("utf-8")
    filename = f"conversation_{chat_id}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    await update.message.reply_document(  # type: ignore[union-attr]
        document=io.BytesIO(content),
        filename=filename,
        caption="📄 Your conversation export",
    )


# ── Admin commands ─────────────────────────────────────────────────────────────

async def cmd_ban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/ban <user_id> — admin only: ban a user."""
    if not is_admin(update.effective_user.id):  # type: ignore[union-attr]
        await update.message.reply_text("⛔ Admins only.")  # type: ignore[union-attr]
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /ban <user_id>")  # type: ignore[union-attr]
        return
    try:
        target = int(args[0])
        await ban_user(target, True)
        await update.message.reply_text(f"✅ User {target} has been banned.")  # type: ignore[union-attr]
    except ValueError:
        await update.message.reply_text("Invalid user_id.")  # type: ignore[union-attr]


async def cmd_unban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/unban <user_id> — admin only: lift a ban."""
    if not is_admin(update.effective_user.id):  # type: ignore[union-attr]
        await update.message.reply_text("⛔ Admins only.")  # type: ignore[union-attr]
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /unban <user_id>")  # type: ignore[union-attr]
        return
    try:
        target = int(args[0])
        await ban_user(target, False)
        await update.message.reply_text(f"✅ User {target} has been unbanned.")  # type: ignore[union-attr]
    except ValueError:
        await update.message.reply_text("Invalid user_id.")  # type: ignore[union-attr]


async def cmd_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/cleanup — admin: remove old messages."""
    if not is_admin(update.effective_user.id):  # type: ignore[union-attr]
        await update.message.reply_text("⛔ Admins only.")  # type: ignore[union-attr]
        return
    deleted = await cleanup_old_conversations()
    await update.message.reply_text(f"🗑️ Cleaned up {deleted} old messages (>{CLEANUP_DAYS} days).")  # type: ignore[union-attr]


async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/broadcast <message> — admin: message all users."""
    if not is_admin(update.effective_user.id):  # type: ignore[union-attr]
        await update.message.reply_text("⛔ Admins only.")  # type: ignore[union-attr]
        return
    text = " ".join(context.args or [])
    if not text:
        await update.message.reply_text("Usage: /broadcast <message>")  # type: ignore[union-attr]
        return
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT user_id FROM users WHERE is_banned=0") as cur:
            user_ids = [r[0] async for r in cur]
    sent = 0
    for uid in user_ids:
        try:
            await context.bot.send_message(uid, text)
            sent += 1
            await asyncio.sleep(0.05)   # respect Telegram rate limits
        except Exception:
            pass
    await update.message.reply_text(f"📢 Broadcast sent to {sent}/{len(user_ids)} users.")  # type: ignore[union-attr]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN MESSAGE HANDLER  (text, documents, images, voice)
# ══════════════════════════════════════════════════════════════════════════════

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Central handler for all non-command messages.
    Dispatches to specialised sub-handlers based on content type.
    """
    if not await guard(update, context):
        return

    message: Message = update.effective_message  # type: ignore[assignment]
    chat_id: int = update.effective_chat.id  # type: ignore[union-attr]
    user_id: int = update.effective_user.id  # type: ignore[union-attr]
    session: aiohttp.ClientSession = context.bot_data["session"]

    # Determine the user content (text, file extraction, voice, image)
    user_content: Any = None
    user_text_for_db: str = ""

    if message.voice:
        user_content, user_text_for_db = await _process_voice(message, session)
    elif message.document:
        user_content, user_text_for_db = await _process_document(message, context)
    elif message.photo:
        user_content, user_text_for_db = await _process_photo(message, context)
    elif message.text:
        user_content = message.text
        user_text_for_db = message.text
    else:
        await message.reply_text("I can handle text, images, voice, PDF, DOCX, and TXT files.")
        return

    if not user_content:
        await message.reply_text("⚠️ Could not process your message.")
        return

    # Persist user message
    await add_message(chat_id, user_id, "user", user_text_for_db)

    # Maybe summarise before adding more
    await maybe_summarise(session, chat_id, user_id)

    # Get preferred model
    user_row = await get_user(user_id)
    model = (user_row or {}).get("preferred_model") or DEFAULT_MODEL

    # Send typing indicator while waiting
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

    # Build messages array
    messages = await build_messages(chat_id, user_id, user_content)

    # Call AI
    t0 = time.perf_counter()
    try:
        reply, model_used, p_tokens, c_tokens = await chat_completion(
            session, messages, model=model, stream=True
        )
    except Exception as exc:
        logger.error("AI call failed: %s\n%s", exc, traceback.format_exc())
        await message.reply_text(
            "❌ All models are currently unavailable. Please try again in a moment."
        )
        return
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if not reply.strip():
        await message.reply_text("⚠️ The model returned an empty response.")
        return

    # Persist assistant response
    await add_message(chat_id, user_id, "assistant", reply, tokens=c_tokens)

    # Record stat
    await record_stat(user_id, chat_id, model_used, p_tokens, c_tokens, latency_ms)

    # Send reply
    await safe_reply(message, reply)


# ── Sub-handlers for specific content types ───────────────────────────────────

async def _process_voice(
    message: Message, session: aiohttp.ClientSession
) -> tuple[str, str]:
    """Download and transcribe a voice message."""
    await message.reply_text("🎤 Transcribing voice message…")
    voice: Voice = message.voice  # type: ignore[assignment]
    data = await download_file(message.get_bot(), voice.file_id)
    transcript = await transcribe_voice(session, data, mime="audio/ogg")
    display = f"[Voice]: {transcript}"
    return display, display


async def _process_document(
    message: Message, context: ContextTypes.DEFAULT_TYPE
) -> tuple[str, str]:
    """Extract text from a document (PDF / DOCX / TXT)."""
    doc: Document = message.document  # type: ignore[assignment]
    fname = (doc.file_name or "").lower()
    mime = doc.mime_type or mimetypes.guess_type(fname)[0] or ""

    await message.reply_text(f"📄 Processing `{doc.file_name}`…", parse_mode=ParseMode.MARKDOWN)
    data = await download_file(context.bot, doc.file_id)

    if fname.endswith(".pdf") or "pdf" in mime:
        extracted = extract_pdf_text(data)
        prefix = "PDF document content"
    elif fname.endswith(".docx") or "word" in mime or "officedocument" in mime:
        extracted = extract_docx_text(data)
        prefix = "DOCX document content"
    elif fname.endswith(".txt") or "text" in mime:
        extracted = extract_txt_text(data)
        prefix = "Text file content"
    else:
        return (
            f"Unsupported file type: {fname or mime}",
            f"Unsupported file type: {fname or mime}",
        )

    caption = message.caption or "Please summarise and analyse this document."
    full = f"[{prefix}]\n{extracted}\n\n[User instruction]: {caption}"
    return full, f"[File: {doc.file_name}] {caption}"


async def _process_photo(
    message: Message, context: ContextTypes.DEFAULT_TYPE
) -> tuple[Any, str]:
    """Build a multimodal message payload for an image."""
    # Get the highest-resolution photo variant
    photo = message.photo[-1]
    data = await download_file(context.bot, photo.file_id)
    mime = "image/jpeg"
    data_url = image_bytes_to_base64_url(data, mime)
    caption = message.caption or "What is in this image?"

    # OpenAI-compatible vision format
    content = [
        {"type": "text", "text": caption},
        {
            "type": "image_url",
            "image_url": {"url": data_url},
        },
    ]
    return content, f"[Image] {caption}"


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log all unhandled errors and optionally notify the user."""
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ An unexpected error occurred. Please try again."
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND TASKS
# ══════════════════════════════════════════════════════════════════════════════

async def background_model_refresh(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodically refresh the list of free models."""
    session: aiohttp.ClientSession = context.bot_data["session"]
    await fetch_free_models(session)
    logger.debug("Background: free-model list refreshed")


async def background_cleanup(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodically remove old conversation rows."""
    deleted = await cleanup_old_conversations(CLEANUP_DAYS)
    if deleted:
        logger.info("Background cleanup: removed %d old messages", deleted)


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

async def post_init(application: Application) -> None:
    """Run after the Application is built but before polling starts."""
    # Set bot commands visible in the Telegram UI
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Welcome message"),
            BotCommand("help", "Show all commands"),
            BotCommand("new", "Start a new conversation"),
            BotCommand("clear", "Clear conversation history"),
            BotCommand("history", "View or search history"),
            BotCommand("stats", "Your usage statistics"),
            BotCommand("ping", "Check bot latency"),
            BotCommand("model", "View/change AI model"),
            BotCommand("models", "List free models"),
            BotCommand("export", "Export conversation"),
        ]
    )
    logger.info("Bot commands registered")

    # Create a shared aiohttp session stored in bot_data
    session = aiohttp.ClientSession()
    application.bot_data["session"] = session

    # Initialise DB
    await init_db()

    # Pre-fetch free models
    await fetch_free_models(session)
    logger.info("Startup complete. Bot is ready.")


async def post_shutdown(application: Application) -> None:
    """Clean up on shutdown."""
    session: aiohttp.ClientSession | None = application.bot_data.get("session")
    if session and not session.closed:
        await session.close()
    logger.info("aiohttp session closed. Goodbye!")


def main() -> None:
    """Entry point — build the Application and start polling."""
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # ── Command handlers ──────────────────────────────────────────────────────
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("help",      cmd_help))
    app.add_handler(CommandHandler("new",       cmd_new))
    app.add_handler(CommandHandler("clear",     cmd_clear))
    app.add_handler(CommandHandler("history",   cmd_history))
    app.add_handler(CommandHandler("stats",     cmd_stats))
    app.add_handler(CommandHandler("ping",      cmd_ping))
    app.add_handler(CommandHandler("model",     cmd_model))
    app.add_handler(CommandHandler("models",    cmd_models))
    app.add_handler(CommandHandler("export",    cmd_export))

    # Admin-only commands
    app.add_handler(CommandHandler("ban",       cmd_ban))
    app.add_handler(CommandHandler("unban",     cmd_unban))
    app.add_handler(CommandHandler("cleanup",   cmd_cleanup))
    app.add_handler(CommandHandler("broadcast", cmd_broadcast))

    # ── Inline keyboard callbacks ─────────────────────────────────────────────
    app.add_handler(CallbackQueryHandler(callback_clear, pattern=r"^clear_"))

    # ── Message handlers (order matters: most specific first) ─────────────────
    app.add_handler(
        MessageHandler(filters.VOICE, handle_message)
    )
    app.add_handler(
        MessageHandler(filters.Document.ALL, handle_message)
    )
    app.add_handler(
        MessageHandler(filters.PHOTO, handle_message)
    )
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    # ── Error handler ─────────────────────────────────────────────────────────
    app.add_error_handler(error_handler)

    # ── Background jobs ───────────────────────────────────────────────────────
    jq = app.job_queue
    if jq:
        jq.run_repeating(background_model_refresh, interval=3600, first=300)
        jq.run_repeating(background_cleanup,       interval=86400, first=3600)

    logger.info("Starting Telegram bot (polling)…")
    app.run_polling(
        poll_interval=1.0,
        timeout=30,
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
