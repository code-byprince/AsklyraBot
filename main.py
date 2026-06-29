"""
Telegram AI Assistant - Production-Ready Bot
Uses OpenRouter API for AI responses, SQLite for persistence.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
import aiosqlite
import docx
from PIL import Image
import pytesseract
try:
    from groq import Groq as _Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
from pydub import AudioSegment
from telegram import (
    BotCommand,
    Chat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
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
from telegram.error import TelegramError

# ─── Environment ────────────────────────────────────────────────────────────
BOT_TOKEN: str = os.environ["BOT_TOKEN"]
OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
ADMIN_IDS_RAW: str = os.environ.get("ADMIN_IDS", "")
ADMIN_IDS: list[int] = [
    int(x.strip()) for x in ADMIN_IDS_RAW.split(",") if x.strip().isdigit()
]
DB_PATH: str = os.environ.get("DB_PATH", "assistant.db")
LOG_DIR: str = os.environ.get("LOG_DIR", "logs")
DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "openrouter/auto")
MAX_HISTORY: int = int(os.environ.get("MAX_HISTORY", "40"))
RATE_LIMIT_SECONDS: int = int(os.environ.get("RATE_LIMIT_SECONDS", "3"))
MAX_CONTEXT_CHARS: int = int(os.environ.get("MAX_CONTEXT_CHARS", "12000"))
APP_URL: str = os.environ.get("APP_URL", "https://github.com/Prince")

# ─── Logging ─────────────────────────────────────────────────────────────────
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = RotatingFileHandler(
    f"{LOG_DIR}/bot.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setFormatter(_fmt)
_file_handler.setLevel(logging.INFO)

_err_handler = RotatingFileHandler(
    f"{LOG_DIR}/error.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_err_handler.setFormatter(_fmt)
_err_handler.setLevel(logging.ERROR)

_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_console.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _err_handler, _console])
logger = logging.getLogger("telegram_ai")

# ─── Free model fallback list ────────────────────────────────────────────────
FREE_MODELS: list[str] = [
    "openrouter/auto",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "openchat/openchat-7b:free",
]

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and friendly AI assistant. "
    "You can help with coding, math, writing, translation, analysis, and general knowledge. "
    "Be concise but thorough. Format code with proper markdown code blocks. "
    "Use Markdown for structure when helpful. "
    "If asked about your identity, say you are an AI assistant powered by OpenRouter."
)

# ─── Rate-limit state (in-memory) ───────────────────────────────────────────
_last_request: dict[int, float] = {}

# ─── Database ────────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables and run auto-migrations."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS users (
                user_id     INTEGER PRIMARY KEY,
                username    TEXT,
                first_name  TEXT,
                last_name   TEXT,
                language    TEXT DEFAULT 'en',
                timezone    TEXT DEFAULT 'UTC',
                model       TEXT DEFAULT 'openrouter/auto',
                joined_at   TEXT DEFAULT (datetime('now')),
                last_seen   TEXT DEFAULT (datetime('now')),
                is_banned   INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS groups (
                chat_id     INTEGER PRIMARY KEY,
                title       TEXT,
                ai_enabled  INTEGER DEFAULT 1,
                model       TEXT DEFAULT 'openrouter/auto',
                joined_at   TEXT DEFAULT (datetime('now')),
                last_active TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                chat_id     INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS memory (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                chat_id     INTEGER NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                updated_at  TEXT DEFAULT (datetime('now')),
                UNIQUE(user_id, chat_id, key)
            );

            CREATE TABLE IF NOT EXISTS settings (
                user_id     INTEGER NOT NULL,
                chat_id     INTEGER NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                PRIMARY KEY (user_id, chat_id, key)
            );

            CREATE TABLE IF NOT EXISTS analytics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER,
                chat_id     INTEGER,
                event_type  TEXT NOT NULL,
                payload     TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics(event_type, created_at DESC);
            """
        )
        await db.commit()
    logger.info("Database initialised at %s", DB_PATH)


async def upsert_user(update: Update) -> None:
    """Insert or update user record."""
    u = update.effective_user
    if not u:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO users (user_id, username, first_name, last_name, last_seen)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                username=excluded.username,
                first_name=excluded.first_name,
                last_name=excluded.last_name,
                last_seen=datetime('now')
            """,
            (u.id, u.username, u.first_name, u.last_name),
        )
        await db.commit()


async def upsert_group(chat: Chat) -> None:
    """Insert or update group record."""
    if chat.type not in ("group", "supergroup"):
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO groups (chat_id, title, last_active)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(chat_id) DO UPDATE SET
                title=excluded.title,
                last_active=datetime('now')
            """,
            (chat.id, chat.title or ""),
        )
        await db.commit()


async def is_banned(user_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT is_banned FROM users WHERE user_id=?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            return bool(row and row[0])


async def get_user_model(user_id: int, chat_id: int) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        # Check group setting first
        if chat_id != user_id:
            async with db.execute(
                "SELECT model FROM groups WHERE chat_id=?", (chat_id,)
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return row[0] or DEFAULT_MODEL
        async with db.execute(
            "SELECT model FROM users WHERE user_id=?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else DEFAULT_MODEL


async def set_user_model(user_id: int, model: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET model=? WHERE user_id=?", (model, user_id)
        )
        await db.commit()


async def get_history(user_id: int, chat_id: int) -> list[dict[str, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT role, content FROM messages
            WHERE user_id=? AND chat_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, chat_id, MAX_HISTORY),
        ) as cur:
            rows = await cur.fetchall()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


async def save_message(
    user_id: int, chat_id: int, role: str, content: str
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages (user_id, chat_id, role, content) VALUES (?,?,?,?)",
            (user_id, chat_id, role, content),
        )
        await db.commit()


async def clear_history(user_id: int, chat_id: int) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM messages WHERE user_id=? AND chat_id=?",
            (user_id, chat_id),
        )
        await db.commit()
        return cur.rowcount


async def get_memory(user_id: int, chat_id: int) -> dict[str, str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT key, value FROM memory WHERE user_id=? AND chat_id=?",
            (user_id, chat_id),
        ) as cur:
            rows = await cur.fetchall()
    return {r[0]: r[1] for r in rows}


async def set_memory(user_id: int, chat_id: int, key: str, value: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO memory (user_id, chat_id, key, value, updated_at)
            VALUES (?,?,?,?,datetime('now'))
            ON CONFLICT(user_id, chat_id, key) DO UPDATE SET
                value=excluded.value, updated_at=datetime('now')
            """,
            (user_id, chat_id, key, value),
        )
        await db.commit()


async def clear_memory(user_id: int, chat_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM memory WHERE user_id=? AND chat_id=?",
            (user_id, chat_id),
        )
        await db.commit()


async def log_event(
    user_id: Optional[int],
    chat_id: Optional[int],
    event_type: str,
    payload: Optional[dict] = None,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO analytics (user_id, chat_id, event_type, payload) VALUES (?,?,?,?)",
            (user_id, chat_id, event_type, json.dumps(payload) if payload else None),
        )
        await db.commit()


async def get_stats() -> dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM users") as cur:
            users = (await cur.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM groups") as cur:
            groups = (await cur.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM messages") as cur:
            msgs = (await cur.fetchone())[0]
        async with db.execute(
            "SELECT COUNT(*) FROM analytics WHERE event_type='ai_request'"
        ) as cur:
            ai_req = (await cur.fetchone())[0]
    return {"users": users, "groups": groups, "messages": msgs, "ai_requests": ai_req}


async def export_history(user_id: int, chat_id: int) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT role, content, created_at FROM messages
            WHERE user_id=? AND chat_id=?
            ORDER BY created_at ASC
            """,
            (user_id, chat_id),
        ) as cur:
            rows = await cur.fetchall()
    lines = [f"[{r[2]}] {r[0].upper()}: {r[1]}" for r in rows]
    return "\n\n".join(lines) if lines else "No history found."


async def is_ai_enabled_in_group(chat_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT ai_enabled FROM groups WHERE chat_id=?", (chat_id,)
        ) as cur:
            row = await cur.fetchone()
            return bool(row is None or row[0])


async def set_group_ai(chat_id: int, enabled: bool) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO groups (chat_id, ai_enabled) VALUES (?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET ai_enabled=excluded.ai_enabled
            """,
            (chat_id, int(enabled)),
        )
        await db.commit()


async def get_all_user_ids() -> list[int]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT user_id FROM users WHERE is_banned=0"
        ) as cur:
            rows = await cur.fetchall()
    return [r[0] for r in rows]


async def get_user_timezone(user_id: int) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT timezone FROM users WHERE user_id=?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else "UTC"


async def set_user_timezone(user_id: int, tz: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET timezone=? WHERE user_id=?", (tz, user_id)
        )
        await db.commit()


# ─── OpenRouter AI ──────────────────────────────────────────────────────────

async def call_openrouter(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    attempt: int = 0,
) -> str:
    """
    Call the OpenRouter API.  Automatically retries with fallback models.
    Returns the assistant reply text or raises after exhausting all models.
    """
    models_to_try = [model] + [m for m in FREE_MODELS if m != model]

    last_error: str = "Unknown error"
    timeout = aiohttp.ClientTimeout(total=60, connect=10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for m in models_to_try:
            try:
                payload = {
                    "model": m,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "stream": False,
                }
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": APP_URL,
                    "X-Title": "Prince Telegram AI Bot",
                }
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                            .strip()
                        )
                        if text:
                            logger.info("AI response via model=%s len=%d", m, len(text))
                            return text
                        last_error = "Empty response from model"
                    elif resp.status == 429:
                        logger.warning("Rate limited on model=%s, trying next", m)
                        await asyncio.sleep(2)
                        last_error = "Rate limited"
                    elif resp.status in (401, 403):
                        body = await resp.text()
                        raise PermissionError(f"Auth error {resp.status}: {body}")
                    else:
                        body = await resp.text()
                        last_error = f"HTTP {resp.status}: {body[:200]}"
                        logger.warning("model=%s status=%d", m, resp.status)
            except PermissionError:
                raise
            except asyncio.TimeoutError:
                last_error = "Request timed out"
                logger.warning("Timeout for model=%s", m)
            except aiohttp.ClientError as exc:
                last_error = f"Network error: {exc}"
                logger.warning("Network error for model=%s: %s", m, exc)
            except Exception as exc:
                last_error = str(exc)
                logger.exception("Unexpected error for model=%s", m)
            await asyncio.sleep(1)

    raise RuntimeError(f"All models failed. Last error: {last_error}")


async def build_messages(
    user_id: int,
    chat_id: int,
    user_text: str,
    image_b64: Optional[str] = None,
) -> list[dict]:
    """Build the full messages list for the API call."""
    history = await get_history(user_id, chat_id)
    mem = await get_memory(user_id, chat_id)

    system_content = SYSTEM_PROMPT
    if mem:
        mem_str = "\n".join(f"- {k}: {v}" for k, v in mem.items())
        system_content += f"\n\nUser memory notes:\n{mem_str}"

    # Truncate history to fit context window
    total_chars = len(system_content) + len(user_text)
    trimmed: list[dict] = []
    for msg in reversed(history):
        total_chars += len(msg["content"])
        if total_chars > MAX_CONTEXT_CHARS:
            break
        trimmed.insert(0, msg)

    messages: list[dict] = [{"role": "system", "content": system_content}]
    messages.extend(trimmed)

    if image_b64:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": user_text or "Describe this image."},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": user_text})

    return messages


# ─── Text helpers ─────────────────────────────────────────────────────────────

_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"


def escape_md(text: str) -> str:
    """Escape special characters for MarkdownV2."""
    for ch in _ESCAPE_CHARS:
        text = text.replace(ch, f"\\{ch}")
    return text


def smart_split(text: str, max_len: int = 4000) -> list[str]:
    """Split a long message into chunks without breaking words/code blocks."""
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    while len(text) > max_len:
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        parts.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        parts.append(text)
    return parts


def format_ai_response(text: str) -> str:
    """
    Try to send as MarkdownV2; return plain text escaped version as fallback.
    Returns the text ready for parse_mode=MarkdownV2.
    """
    # We keep code blocks and inline code intact; escape the rest.
    # Strategy: split on code blocks, escape non-code segments.
    parts = re.split(r"(```[\s\S]*?```|`[^`]+`)", text)
    result = []
    for part in parts:
        if part.startswith("```") or (part.startswith("`") and part.endswith("`")):
            result.append(part)
        else:
            result.append(escape_md(part))
    return "".join(result)


# ─── Rate limiting ────────────────────────────────────────────────────────────

def check_rate_limit(user_id: int) -> bool:
    """Returns True if user is within rate limit (allowed). Updates state."""
    now = time.monotonic()
    last = _last_request.get(user_id, 0.0)
    if now - last < RATE_LIMIT_SECONDS:
        return False
    _last_request[user_id] = now
    return True


# ─── File readers ─────────────────────────────────────────────────────────────

async def read_docx(file_bytes: bytes) -> str:
    """Extract text from a .docx file."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = docx.Document(tmp_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def read_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF using pdfminer."""
    try:
        from pdfminer.high_level import extract_text as pm_extract
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            text = pm_extract(tmp_path)
            return text.strip() if text else "[No text extracted from PDF]"
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except ImportError:
        return "[PDF reading requires pdfminer.six — install it]"


async def ocr_image(image_bytes: bytes) -> str:
    """Run OCR on an image and return extracted text."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip() if text.strip() else "[No text found in image via OCR]"
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return f"[OCR failed: {exc}]"


async def transcribe_audio(ogg_bytes: bytes) -> str:
    """Convert OGG voice to MP3 and transcribe using Groq Whisper API."""
    if not _GROQ_AVAILABLE or not GROQ_API_KEY:
        return "[Voice transcription unavailable: set GROQ_API_KEY env variable]"
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f_ogg:
            f_ogg.write(ogg_bytes)
            ogg_path = f_ogg.name
        mp3_path = ogg_path.replace(".ogg", ".mp3")
        try:
            audio = AudioSegment.from_ogg(ogg_path)
            audio.export(mp3_path, format="mp3")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _run_groq_transcribe, mp3_path)
            return result
        finally:
            Path(ogg_path).unlink(missing_ok=True)
            Path(mp3_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Audio transcription failed: %s", exc)
        return f"[Audio processing failed: {exc}]"


def _run_groq_transcribe(mp3_path: str) -> str:
    """Call Groq Whisper API synchronously (runs in executor)."""
    try:
        client = _Groq(api_key=GROQ_API_KEY)
        with open(mp3_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(Path(mp3_path).name, f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )
        return str(transcription).strip() if transcription else "[Could not understand the audio]"
    except Exception as exc:
        return f"[Groq transcription error: {exc}]"


# ─── Core AI handler ──────────────────────────────────────────────────────────

async def handle_ai_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
    image_b64: Optional[str] = None,
) -> None:
    """Send a message to OpenRouter and reply."""
    user = update.effective_user
    chat = update.effective_chat
    msg = update.effective_message

    if not user or not chat or not msg:
        return

    if await is_banned(user.id):
        return

    if not check_rate_limit(user.id):
        await msg.reply_text("⏳ Please wait a moment before sending another message.")
        return

    await upsert_user(update)
    if chat.type in ("group", "supergroup"):
        await upsert_group(chat)

    # Typing indicator
    await context.bot.send_chat_action(chat.id, ChatAction.TYPING)

    model = await get_user_model(user.id, chat.id)
    messages = await build_messages(user.id, chat.id, user_text, image_b64)

    await log_event(user.id, chat.id, "ai_request", {"model": model})

    try:
        reply = await call_openrouter(messages, model)
    except PermissionError as exc:
        logger.error("Auth error: %s", exc)
        await msg.reply_text("❌ API authentication failed. Please check OPENROUTER_API_KEY.")
        return
    except RuntimeError as exc:
        logger.error("AI call failed: %s", exc)
        await msg.reply_text(
            "❌ The AI service is temporarily unavailable. Please try again in a moment."
        )
        return

    await save_message(user.id, chat.id, "user", user_text)
    await save_message(user.id, chat.id, "assistant", reply)

    # Send reply in chunks
    chunks = smart_split(reply)
    for i, chunk in enumerate(chunks):
        try:
            formatted = format_ai_response(chunk)
            await msg.reply_text(formatted, parse_mode=ParseMode.MARKDOWN_V2)
        except TelegramError:
            # Fallback to plain text
            try:
                await msg.reply_text(chunk)
            except TelegramError as exc:
                logger.error("Failed to send message chunk: %s", exc)


# ─── Command handlers ─────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return
    await upsert_user(update)
    await log_event(user.id, update.effective_chat.id if update.effective_chat else None, "start")
    name = user.first_name or "there"
    text = (
        f"👋 *Hello, {escape_md(name)}\\!*\n\n"
        "I'm your AI assistant powered by OpenRouter\\. "
        "I can help with coding, writing, math, translation, and much more\\.\n\n"
        "*What I can do:*\n"
        "💬 Chat naturally in any language\n"
        "🖼 Analyze images\n"
        "🎤 Transcribe voice messages\n"
        "📄 Read PDF, DOCX, TXT files\n"
        "💻 Generate and explain code\n"
        "🧮 Solve math problems\n\n"
        "Type `/help` to see all commands\\."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "📖 *Commands*\n\n"
        "/start — Start the bot\n"
        "/help — Show this help\n"
        "/new — Start a new conversation\n"
        "/clear — Clear chat history\n"
        "/reset — Reset everything \\(history \\+ memory\\)\n"
        "/history — Show recent messages\n"
        "/export — Export your conversation\n"
        "/stats — Your usage statistics\n"
        "/ping — Check bot status\n"
        "/model — Show current AI model\n"
        "/models — List available models\n"
        "/id — Show your Telegram ID\n"
        "/about — About this bot\n"
        "/settings — Manage your settings\n\n"
        "📎 *File support:* PDF, DOCX, TXT, images, voice\n"
        "💬 *Groups:* Mention me to get a reply"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return
    count = await clear_history(user.id, chat.id)
    await update.message.reply_text(
        f"✅ New conversation started. Cleared {count} messages."
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_new(update, context)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return
    await clear_history(user.id, chat.id)
    await clear_memory(user.id, chat.id)
    await update.message.reply_text("✅ History and memory fully reset.")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return
    history = await get_history(user.id, chat.id)
    if not history:
        await update.message.reply_text("📭 No conversation history yet.")
        return
    lines = []
    for msg in history[-10:]:
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        snippet = msg["content"][:100].replace("\n", " ")
        lines.append(f"{role_icon} {msg['role'].upper()}: {snippet}…")
    await update.message.reply_text("\n\n".join(lines))


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return
    content = await export_history(user.id, chat.id)
    if content == "No history found.":
        await update.message.reply_text("📭 Nothing to export yet.")
        return
    buf = io.BytesIO(content.encode("utf-8"))
    buf.name = f"conversation_{user.id}_{chat.id}.txt"
    await update.message.reply_document(buf, filename=buf.name, caption="📄 Your conversation export")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id=? AND chat_id=?",
            (user.id, chat.id),
        ) as cur:
            msg_count = (await cur.fetchone())[0]
        async with db.execute(
            "SELECT COUNT(*) FROM analytics WHERE user_id=? AND event_type='ai_request'",
            (user.id,),
        ) as cur:
            ai_count = (await cur.fetchone())[0]
    tz_str = await get_user_timezone(user.id)
    model = await get_user_model(user.id, chat.id)
    text = (
        f"📊 *Your Stats*\n\n"
        f"Messages in this chat: `{msg_count}`\n"
        f"Total AI requests: `{ai_count}`\n"
        f"Current model: `{escape_md(model)}`\n"
        f"Timezone: `{escape_md(tz_str)}`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    start = time.monotonic()
    msg = await update.message.reply_text("🏓 Pong\\!", parse_mode=ParseMode.MARKDOWN_V2)
    elapsed = (time.monotonic() - start) * 1000
    await msg.edit_text(
        f"🏓 Pong\\! Response time: `{elapsed:.0f}ms`",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return
    model = await get_user_model(user.id, chat.id)
    await update.message.reply_text(
        f"🤖 Current model: `{escape_md(model)}`",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["🤖 *Available Models*\n"]
    for m in FREE_MODELS:
        lines.append(f"• `{escape_md(m)}`")
    lines.append("\nUse `/settings` to change your model\\.")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return
    text = (
        f"🪪 *Your IDs*\n\n"
        f"User ID: `{user.id}`\n"
        f"Chat ID: `{chat.id}`\n"
        f"Username: `{escape_md(user.username or 'none')}`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🤖 *Telegram AI Assistant*\n\n"
        "Powered by *OpenRouter* with automatic model fallback\\.\n"
        "Built with *python\\-telegram\\-bot v22\\+*\\.\n\n"
        "Features: multi\\-language, voice, images, PDFs, code, math, memory\n\n"
        f"Source: `{escape_md(APP_URL)}`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return
    model = await get_user_model(user.id, chat.id)
    tz = await get_user_timezone(user.id)

    keyboard = [
        [InlineKeyboardButton("🤖 Change Model", callback_data="settings:model")],
        [InlineKeyboardButton("🌍 Set Timezone", callback_data="settings:timezone")],
        [InlineKeyboardButton("🗑 Clear Memory", callback_data="settings:clearmem")],
        [InlineKeyboardButton("📤 Export Chat", callback_data="settings:export")],
    ]
    text = (
        f"⚙️ *Settings*\n\n"
        f"Model: `{escape_md(model)}`\n"
        f"Timezone: `{escape_md(tz)}`"
    )
    await update.message.reply_text(
        text,
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return

    action = query.data.split(":")[1] if ":" in query.data else ""

    if action == "model":
        buttons = [
            [InlineKeyboardButton(m[:40], callback_data=f"setmodel:{m}")]
            for m in FREE_MODELS
        ]
        buttons.append([InlineKeyboardButton("◀ Back", callback_data="settings:back")])
        await query.edit_message_text(
            "🤖 *Select a model:*",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=InlineKeyboardMarkup(buttons),
        )
    elif action == "timezone":
        await query.edit_message_text(
            "🌍 Send your timezone as text \\(e\\.g\\. `Asia/Kolkata`, `America/New_York`\\)\\.\n\n"
            "Reply to this message with your timezone\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        context.user_data["awaiting_timezone"] = True
    elif action == "clearmem":
        await clear_memory(user.id, chat.id)
        await query.edit_message_text("✅ Memory cleared.")
    elif action == "export":
        content = await export_history(user.id, chat.id)
        buf = io.BytesIO(content.encode())
        buf.name = f"chat_{user.id}.txt"
        await context.bot.send_document(chat.id, buf, filename=buf.name)
        await query.edit_message_text("✅ Export sent.")
    elif action == "back":
        await cmd_settings(update, context)


async def setmodel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    if not user:
        return
    model = query.data.split(":", 1)[1]
    await set_user_model(user.id, model)
    await query.edit_message_text(
        f"✅ Model set to:\n`{escape_md(model)}`",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


# ─── Admin commands ────────────────────────────────────────────────────────────

def admin_only(func):
    """Decorator: restrict command to admin users."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user or user.id not in ADMIN_IDS:
            await update.message.reply_text("⛔ Admin only.")
            return
        return await func(update, context)
    return wrapper


@admin_only
async def cmd_admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    stats = await get_stats()
    text = (
        f"📊 *Bot Analytics*\n\n"
        f"Users: `{stats['users']}`\n"
        f"Groups: `{stats['groups']}`\n"
        f"Messages: `{stats['messages']}`\n"
        f"AI Requests: `{stats['ai_requests']}`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


@admin_only
async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /broadcast <message>")
        return
    text = " ".join(context.args)
    user_ids = await get_all_user_ids()
    sent = 0
    failed = 0
    for uid in user_ids:
        try:
            await context.bot.send_message(uid, text)
            sent += 1
            await asyncio.sleep(0.05)
        except TelegramError:
            failed += 1
    await update.message.reply_text(f"📢 Broadcast done. Sent: {sent}, Failed: {failed}")


@admin_only
async def cmd_ban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /ban <user_id>")
        return
    try:
        target = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid user ID.")
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE users SET is_banned=1 WHERE user_id=?", (target,))
        await db.commit()
    await update.message.reply_text(f"🔨 User {target} banned.")


@admin_only
async def cmd_unban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /unban <user_id>")
        return
    try:
        target = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid user ID.")
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE users SET is_banned=0 WHERE user_id=?", (target,))
        await db.commit()
    await update.message.reply_text(f"✅ User {target} unbanned.")


@admin_only
async def cmd_aienable(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    await set_group_ai(chat.id, True)
    await update.message.reply_text("✅ AI enabled in this group.")


@admin_only
async def cmd_aidisable(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return
    await set_group_ai(chat.id, False)
    await update.message.reply_text("✅ AI disabled in this group.")


@admin_only
async def cmd_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete messages older than 90 days."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM messages WHERE created_at < datetime('now','-90 days')"
        )
        count = cur.rowcount
        await db.commit()
    await update.message.reply_text(f"🗑 Cleaned up {count} old messages.")


# ─── Message handlers ─────────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages."""
    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not msg or not chat or not user:
        return

    # Handle timezone setting flow
    if context.user_data.get("awaiting_timezone"):
        tz_str = msg.text.strip()
        try:
            ZoneInfo(tz_str)
            await set_user_timezone(user.id, tz_str)
            context.user_data.pop("awaiting_timezone", None)
            await msg.reply_text(f"✅ Timezone set to `{escape_md(tz_str)}`", parse_mode=ParseMode.MARKDOWN_V2)
        except (ZoneInfoNotFoundError, KeyError):
            await msg.reply_text("❌ Invalid timezone. Try something like `Asia/Kolkata` or `America/New_York`.")
        return

    # Group: only respond when mentioned or in reply to bot
    if chat.type in ("group", "supergroup"):
        bot_username = context.bot.username
        is_mentioned = (
            msg.text
            and bot_username
            and f"@{bot_username}".lower() in msg.text.lower()
        )
        is_reply_to_bot = (
            msg.reply_to_message
            and msg.reply_to_message.from_user
            and msg.reply_to_message.from_user.id == context.bot.id
        )
        if not is_mentioned and not is_reply_to_bot:
            return
        if not await is_ai_enabled_in_group(chat.id):
            return
        # Strip bot mention from text
        text = re.sub(
            rf"@{re.escape(bot_username or '')}", "", msg.text or "", flags=re.IGNORECASE
        ).strip()
    else:
        text = msg.text or ""

    if not text:
        await msg.reply_text("Please send me a message to respond to.")
        return

    await handle_ai_message(update, context, text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Transcribe voice message then answer."""
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return

    # Group filter
    if chat.type in ("group", "supergroup"):
        if not await is_ai_enabled_in_group(chat.id):
            return

    await context.bot.send_chat_action(chat.id, ChatAction.TYPING)
    voice = msg.voice
    file = await context.bot.get_file(voice.file_id)
    ogg_bytes = await file.download_as_bytearray()
    transcript = await transcribe_audio(bytes(ogg_bytes))
    await msg.reply_text(f"🎤 *Transcript:* {escape_md(transcript)}", parse_mode=ParseMode.MARKDOWN_V2)
    await handle_ai_message(update, context, transcript)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages — send to vision model or OCR."""
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return

    if chat.type in ("group", "supergroup"):
        bot_username = context.bot.username
        is_mentioned = bool(
            msg.caption
            and bot_username
            and f"@{bot_username}".lower() in (msg.caption or "").lower()
        )
        is_reply_to_bot = bool(
            msg.reply_to_message
            and msg.reply_to_message.from_user
            and msg.reply_to_message.from_user.id == context.bot.id
        )
        if not is_mentioned and not is_reply_to_bot:
            return
        if not await is_ai_enabled_in_group(chat.id):
            return

    photo = msg.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    img_bytes = await file.download_as_bytearray()
    img_b64 = base64.b64encode(bytes(img_bytes)).decode()
    caption = msg.caption or "Describe this image in detail."
    await handle_ai_message(update, context, caption, image_b64=img_b64)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads: PDF, DOCX, TXT."""
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return

    if chat.type in ("group", "supergroup"):
        if not await is_ai_enabled_in_group(chat.id):
            return

    doc = msg.document
    if not doc:
        return

    fname = doc.file_name or ""
    ext = Path(fname).suffix.lower()

    if ext not in (".pdf", ".docx", ".txt", ".md", ".csv"):
        await msg.reply_text(
            "📎 I can read PDF, DOCX, TXT, MD, and CSV files. Please send one of those."
        )
        return

    await context.bot.send_chat_action(chat.id, ChatAction.TYPING)
    file = await context.bot.get_file(doc.file_id)
    file_bytes = await file.download_as_bytearray()

    if ext == ".pdf":
        content = await read_pdf(bytes(file_bytes))
    elif ext == ".docx":
        content = await read_docx(bytes(file_bytes))
    else:
        try:
            content = bytes(file_bytes).decode("utf-8", errors="replace")
        except Exception:
            content = "[Could not decode file]"

    caption = msg.caption or f"Analyze this {ext.upper()} document and summarize it."
    combined = f"[Document: {fname}]\n\n{content[:MAX_CONTEXT_CHARS]}\n\n---\nUser request: {caption}"
    await handle_ai_message(update, context, combined)


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler — log and recover."""
    logger.error(
        "Exception while handling update:\n%s",
        "".join(traceback.format_exception(type(context.error), context.error, context.error.__traceback__))
        if context.error
        else "No error info",
    )
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ An internal error occurred. Please try again."
            )
        except Exception:
            pass


# ─── Bot setup ────────────────────────────────────────────────────────────────

async def post_init(app: Application) -> None:
    """Set bot commands after startup."""
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Show help"),
        BotCommand("new", "Start new conversation"),
        BotCommand("clear", "Clear chat history"),
        BotCommand("reset", "Reset history and memory"),
        BotCommand("history", "View recent messages"),
        BotCommand("export", "Export conversation"),
        BotCommand("stats", "Your usage stats"),
        BotCommand("ping", "Check bot status"),
        BotCommand("model", "Show current model"),
        BotCommand("models", "List available models"),
        BotCommand("id", "Show your Telegram ID"),
        BotCommand("about", "About this bot"),
        BotCommand("settings", "Manage settings"),
    ]
    await app.bot.set_my_commands(commands)
    logger.info("Bot commands registered.")


def build_application() -> Application:
    """Build and configure the Application."""
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # User commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("settings", cmd_settings))

    # Admin commands
    app.add_handler(CommandHandler("adminstats", cmd_admin_stats))
    app.add_handler(CommandHandler("broadcast", cmd_broadcast))
    app.add_handler(CommandHandler("ban", cmd_ban))
    app.add_handler(CommandHandler("unban", cmd_unban))
    app.add_handler(CommandHandler("aienable", cmd_aienable))
    app.add_handler(CommandHandler("aidisable", cmd_aidisable))
    app.add_handler(CommandHandler("cleanup", cmd_cleanup))

    # Inline buttons
    app.add_handler(CallbackQueryHandler(settings_callback, pattern=r"^settings:"))
    app.add_handler(CallbackQueryHandler(setmodel_callback, pattern=r"^setmodel:"))

    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    app.add_error_handler(handle_error)

    return app


def main() -> None:
    """Entry point."""
    asyncio.run(init_db())
    logger.info("Starting Telegram AI Assistant…")
    app = build_application()
    logger.info("Bot polling started.")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
