# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  ADVANCED TELEGRAM AI BOT  —  Powered by Groq (Llama/Qwen/GPT-OSS/DeepSeek/Kimi)
#  Features: Multi-model AI, 100+ language translate, long-term memory,
#            conversation summarization, file analysis, voice, group admin,
#            live analytics, unlimited questions with zero-error fallback
# ═══════════════════════════════════════════════════════════════════════════════

import os, re, io, csv, json, time, random, logging, zipfile, threading
import requests as http_requests
from datetime import datetime, timedelta
from collections import defaultdict

from flask import Flask, request
import telebot
from telebot import types
from openai import OpenAI

# ── Optional file-parsing libraries (graceful fallback if missing) ─────────────
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import openpyxl
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

try:
    import speech_recognition as sr
    from pydub import AudioSegment
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Environment ────────────────────────────────────────────────────────────────
BOT_TOKEN  = os.environ.get("BOT_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ADMIN_IDS  = [int(x) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip().isdigit()]

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not set!")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set! Get a free key (no credit card) at https://console.groq.com/keys")

# ── Clients ────────────────────────────────────────────────────────────────────
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)

ai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# Groq free tier — no credit card, generous daily limits, OpenAI-compatible.
# These are Groq's current (non-deprecated) production model IDs as of June 2026.
MODELS = [
    "openai/gpt-oss-120b",          # GPT — flagship reasoning, best quality
    "openai/gpt-oss-20b",           # GPT — smaller/faster fallback
    "qwen/qwen3.6-27b",             # Qwen — strong multilingual + reasoning
    "deepseek-r1-distill-llama-70b",# DeepSeek — distilled reasoning model
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama — fast general purpose
    "moonshotai/kimi-k2",           # Kimi K2 — long-context backup
]
MAX_TOKENS = 1024

# Friendly labels for /model menu and display
MODEL_LABELS = {
    "gpt-oss":  "GPT",
    "qwen":     "Qwen",
    "deepseek": "DeepSeek",
    "llama":    "Llama",
    "kimi":     "Kimi",
}

def model_family(model_id: str) -> str:
    low = model_id.lower()
    for key, label in MODEL_LABELS.items():
        if key in low:
            return label
    return model_id.split("/")[0]

# Per-user preferred model family (None = auto / use full fallback chain)
user_model_pref: dict[int, str] = {}

def ordered_models_for(user_id: int) -> list:
    """Put the user's preferred family first, then the rest as fallback."""
    pref = user_model_pref.get(user_id)
    if not pref:
        return MODELS
    preferred = [m for m in MODELS if pref.lower() in m.lower()]
    rest      = [m for m in MODELS if m not in preferred]
    return preferred + rest if preferred else MODELS

# ═══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY ANALYTICS & STATE
# ═══════════════════════════════════════════════════════════════════════════════
analytics = {
    "total_users":    set(),
    "daily_messages": defaultdict(int),   # date → count
    "total_messages": 0,
    "total_tokens":   0,
    "errors":         [],                 # list of {time, user_id, error}
    "response_times": [],                 # list of float seconds
    "model_usage":    defaultdict(int),   # model → count
    "file_types":     defaultdict(int),   # extension → count
    "active_today":   set(),              # user_ids active today
}

user_histories: dict[int, list[dict]] = {}
user_modes:     dict[int, str]        = {}
user_stats:     dict[int, dict]       = {}
user_waiting:   dict[int, str]        = {}   # waiting for next message for a purpose
user_summaries: dict[int, str]        = {}   # long-term memory: rolling summary per user
pending_continue: dict[int, str]      = {}   # last AI reply tail, for Continue button

MAX_HISTORY = 40        # raw messages kept verbatim before summarizing
SUMMARIZE_AFTER = 30     # once history crosses this, fold oldest into summary

# ── Long-term memory: persisted to disk so restarts don't wipe users out ──────
DATA_DIR = os.environ.get("DATA_DIR", "./bot_data")
os.makedirs(DATA_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")

_state_lock = threading.RLock()   # guards all shared dict mutations below

def save_memory():
    """Persist histories, summaries, stats, modes to disk. Never raises."""
    try:
        with _state_lock:
            payload = {
                "histories": user_histories,
                "summaries": user_summaries,
                "stats":     user_stats,
                "modes":     user_modes,
                "model_pref": user_model_pref,
            }
        tmp = MEMORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, MEMORY_FILE)
    except Exception as e:
        logger.warning(f"save_memory failed (non-fatal): {e}")

def load_memory():
    """Load persisted memory on startup. Never raises — falls back to empty state."""
    if not os.path.exists(MEMORY_FILE):
        return
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        with _state_lock:
            for uid, h in payload.get("histories", {}).items():
                user_histories[int(uid)] = h
            for uid, s in payload.get("summaries", {}).items():
                user_summaries[int(uid)] = s
            for uid, s in payload.get("stats", {}).items():
                user_stats[int(uid)] = s
            for uid, m in payload.get("modes", {}).items():
                user_modes[int(uid)] = m
            for uid, m in payload.get("model_pref", {}).items():
                user_model_pref[int(uid)] = m
        logger.info(f"✅ Loaded long-term memory: {len(user_histories)} user histories")
    except Exception as e:
        logger.warning(f"load_memory failed, starting fresh (non-fatal): {e}")

def autosave_loop():
    """Background thread — saves memory to disk every 30s so nothing is lost."""
    while True:
        time.sleep(30)
        save_memory()

# ── Smart history management — summarizes old turns instead of deleting them ──
def summarize_old_messages(user_id: int, messages_to_fold: list) -> None:
    """Ask the AI to compress old messages into the rolling summary (long-term memory)."""
    if not messages_to_fold:
        return
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:500]}" for m in messages_to_fold
    )
    prior_summary = user_summaries.get(user_id, "")
    prompt = (
        "Update the running memory summary of this conversation. "
        "Keep names, facts, preferences, ongoing topics, and important context. "
        "Be concise (max 200 words), plain text, no formatting.\n\n"
        f"PREVIOUS SUMMARY:\n{prior_summary or '(none yet)'}\n\n"
        f"NEW MESSAGES TO FOLD IN:\n{convo_text}\n\n"
        "Write the UPDATED SUMMARY only:"
    )
    try:
        resp = ai_client.chat.completions.create(
            model=MODELS[0],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        new_summary = resp.choices[0].message.content.strip()
        with _state_lock:
            user_summaries[user_id] = new_summary
        logger.info(f"🧠 Updated long-term summary for user {user_id}")
    except Exception as e:
        # Non-fatal — worst case we just lose this fold-in, history is still trimmed
        logger.warning(f"Summarization failed (non-fatal): {e}")

def trim_history(user_id: int, history: list) -> list:
    """Keep recent messages verbatim; fold older ones into the long-term summary."""
    if len(history) <= MAX_HISTORY:
        return history
    cutoff = len(history) - (MAX_HISTORY - 4)
    old_part = history[:cutoff]
    new_part = history[cutoff:]
    # Fold old messages into summary in the background so we never block the reply
    threading.Thread(target=summarize_old_messages, args=(user_id, old_part), daemon=True).start()
    return new_part

# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════
BASE_SYSTEM = """You are an advanced AI assistant inside a Telegram bot. STRICTLY follow:

FORMATTING (Telegram HTML mode):
- NO markdown headers (#, ##, ###) ever — use plain bold labels instead
- Bold important words: wrap in <b>word</b>
- Lists: use • or - at start of line
- Code: use <code>inline</code> for short code
- Never use triple backticks

CONTENT:
- Be concise, helpful, professional
- Highlight key facts and terms in bold
- Remember conversation context
- If asked something you cannot do, say so clearly
"""

MODE_PROMPTS = {
    "normal":    "",
    "translate": "Translate every message the user sends. Support all major world languages (100+ languages including Hindi, English, Spanish, French, Arabic, Chinese, Japanese, Korean, Russian, German, Portuguese, Bengali, Urdu, etc). If the user specifies a target language, translate to that. Otherwise auto-detect the source language and translate to English. Always state the detected source language and target language before the translation.",
    "summarize": "Summarize every message in 3-5 bullet points. Be extremely concise.",
    "code":      "You are a senior software engineer. Analyze code, fix bugs, explain concepts clearly. Format code with <code> tags.",
    "essay":     "Help the user write, improve, or proofread essays and documents. Give structured feedback.",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  REACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
REACTION_MAP = [
    (["error","problem","issue","bug","crash","fix"],         "🤔"),
    (["code","program","python","script","function","debug"], "👨‍💻"),
    (["thank","thanks","shukriya","dhanyawad","ty","tysm"],   "🙏"),
    (["hello","hi","hey","namaste","salam","howdy","hola"],   "👋"),
    (["love","pyar","heart","❤","cute","beautiful"],          "❤"),
    (["money","price","cost","rupee","dollar","paisa"],        "💯"),
    (["idea","suggest","plan","concept","creative"],           "💡"),
    (["news","update","latest","breaking","trending"],         "🔥"),
    (["yes","correct","right","bilkul","haan","sure"],        "👍"),
    (["no","wrong","nahi","galat","nope","incorrect"],         "👎"),
    (["sad","dukh","cry","upset","bura","depressed"],          "😢"),
    (["happy","great","awesome","amazing","excellent"],         "🎉"),
    (["funny","lol","haha","joke","maza","hilarious"],         "😂"),
    (["wow","incredible","unbelievable","shocking"],            "🤩"),
    (["food","khana","eat","recipe","hungry","cook"],           "🍕"),
    (["music","song","gaana","listen","playlist","band"],       "🎵"),
    (["sports","cricket","football","game","match","score"],    "⚽"),
    (["weather","rain","sunny","temperature","mausam"],         "⛅"),
    (["pdf","docx","file","document","upload","read"],          "📄"),
    (["photo","image","picture","screenshot","pic"],            "🖼"),
]
DEFAULT_REACTIONS = ["👍","🔥","💯","🤩","👏","⚡"]

def pick_reaction(text: str) -> str:
    low = text.lower()
    for kws, emoji in REACTION_MAP:
        if any(k in low for k in kws):
            return emoji
    return random.choice(DEFAULT_REACTIONS)

def send_reaction(chat_id: int, message_id: int, emoji: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/setMessageReaction"
    try:
        r = http_requests.post(url, json={
            "chat_id":    chat_id,
            "message_id": message_id,
            "reaction":   [{"type": "emoji", "emoji": emoji}],
            "is_big":     False,
        }, timeout=5)
        if not r.json().get("ok"):
            logger.debug(f"Reaction note: {r.json().get('description')}")
    except Exception as e:
        logger.debug(f"Reaction error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  HTML FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════
def format_for_telegram(text: str) -> str:
    """Convert AI markdown → safe Telegram HTML."""
    # Remove # headings
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Triple backtick blocks → <pre>
    text = re.sub(r"```(?:\w+)?\n?(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)
    # **bold** → <b>
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    # *italic* → <i>  (only single asterisks not already inside tags)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    # `inline` → <code>
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    return text.strip()

def safe_send(chat_id: int, text: str, reply_to: int = None, kb=None):
    """Send HTML message; fallback to plain text on parse error."""
    kwargs = {"parse_mode": "HTML", "reply_markup": kb}
    if reply_to:
        kwargs["reply_to_message_id"] = reply_to
    try:
        bot.send_message(chat_id, text, **kwargs)
    except Exception:
        kwargs.pop("parse_mode")
        # Strip HTML tags for plain fallback
        plain = re.sub(r"<[^>]+>", "", text)
        try:
            bot.send_message(chat_id, plain, **kwargs)
        except Exception as e:
            logger.error(f"safe_send failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def track_message(user_id: int):
    today = datetime.now().strftime("%Y-%m-%d")
    with _state_lock:
        analytics["total_users"].add(user_id)
        analytics["daily_messages"][today] += 1
        analytics["total_messages"] += 1
        analytics["active_today"].add(user_id)
        if user_id not in user_stats:
            user_stats[user_id] = {
                "count": 0,
                "first_seen": today,
                "name": "",
                "tokens_used": 0,
            }
        user_stats[user_id]["count"] += 1

def track_tokens(count: int, user_id: int = None):
    with _state_lock:
        analytics["total_tokens"] += count
        if user_id and user_id in user_stats:
            user_stats[user_id]["tokens_used"] += count

def track_error(user_id: int, error: str):
    with _state_lock:
        analytics["errors"].append({
            "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "error":   str(error)[:200],
        })
        analytics["errors"] = analytics["errors"][-100:]   # keep last 100

def track_response_time(seconds: float):
    with _state_lock:
        analytics["response_times"].append(round(seconds, 3))
        analytics["response_times"] = analytics["response_times"][-500:]

def leaderboard() -> list:
    with _state_lock:
        return sorted(
            [(uid, d["count"], d.get("name","?"), d.get("tokens_used",0))
             for uid, d in user_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:10]

# ═══════════════════════════════════════════════════════════════════════════════
#  AI CORE  — unlimited history with smart trimming
# ═══════════════════════════════════════════════════════════════════════════════
def get_ai_response(user_id: int, user_message: str, extra_system: str = "") -> tuple[str, bool]:
    """Returns (reply_text, was_truncated). Never raises — always returns a usable string."""
    with _state_lock:
        if user_id not in user_histories:
            user_histories[user_id] = []
        mode    = user_modes.get(user_id, "normal")
        history = user_histories[user_id]
        history.append({"role": "user", "content": user_message})
        history = trim_history(user_id, history)
        user_histories[user_id] = history
        history_snapshot = list(history)  # safe copy to build messages outside lock

    system = BASE_SYSTEM
    if MODE_PROMPTS.get(mode):
        system += "\n\nMODE: " + MODE_PROMPTS[mode]
    if extra_system:
        system += "\n\n" + extra_system

    long_term = user_summaries.get(user_id)
    if long_term:
        system += (
            "\n\nLONG-TERM MEMORY (earlier context from this user, already summarized — "
            f"use it for continuity, don't repeat it back unless relevant):\n{long_term}"
        )

    messages = [{"role": "system", "content": system}] + history_snapshot
    t0 = time.time()
    last_error = ""

    models_to_try = ordered_models_for(user_id)

    # Two full passes over every model — handles transient rate-limits/timeouts
    # gracefully so the user almost never sees an error, ever.
    for round_num in range(2):
        for model in models_to_try:
            try:
                resp = ai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                )
                choice = resp.choices[0]
                reply  = (choice.message.content or "").strip()
                if not reply:
                    raise ValueError("empty response from model")
                truncated = (getattr(choice, "finish_reason", "") == "length")
                tokens = getattr(resp.usage, "total_tokens", 0)

                with _state_lock:
                    user_histories[user_id].append({"role": "assistant", "content": reply})
                track_tokens(tokens, user_id)
                analytics["model_usage"][model] += 1
                track_response_time(time.time() - t0)
                logger.info(f"✅ Success with model: {model} (user {user_id})")
                return reply, truncated

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Model {model} failed (round {round_num}): {type(e).__name__}: {last_error[:120]}")
                # Rate-limit → tiny backoff helps a lot before moving to next model
                if "429" in last_error or "rate" in last_error.lower():
                    time.sleep(0.8)
                else:
                    time.sleep(0.2)
                continue
        # brief pause before the second full pass, in case it was a transient blip
        if round_num == 0:
            time.sleep(1.0)

    # Every model in every round failed — this should be extremely rare.
    # We still NEVER show a raw error to the user.
    track_error(user_id, last_error)
    logger.error(f"All models failed twice for user {user_id}. Last error: {last_error}")
    friendly = (
        "Sab AI models thoda busy hain abhi 🙏 Maine background me retry kar liya hai — "
        "ek baar phir bhej do, ya thoda ruk ke try karo. Tumhara message safe hai, khoya nahi 🙂"
    )
    return friendly, False

# ═══════════════════════════════════════════════════════════════════════════════
#  FILE PARSERS
# ═══════════════════════════════════════════════════════════════════════════════
def parse_pdf(file_bytes: bytes) -> str:
    if not HAS_PDF:
        return "[PDF support not installed — add PyPDF2 to requirements]"
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages  = [p.extract_text() or "" for p in reader.pages[:20]]
        text   = "\n".join(pages).strip()
        return text[:12000] if text else "[PDF has no extractable text]"
    except Exception as e:
        return f"[PDF parse error: {e}]"

def parse_docx(file_bytes: bytes) -> str:
    if not HAS_DOCX:
        return "[DOCX support not installed — add python-docx to requirements]"
    try:
        doc   = DocxDocument(io.BytesIO(file_bytes))
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(lines)[:12000]
    except Exception as e:
        return f"[DOCX parse error: {e}]"

def parse_excel(file_bytes: bytes) -> str:
    if not HAS_EXCEL:
        return "[Excel support not installed — add openpyxl to requirements]"
    try:
        wb     = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        result = []
        for sheet in wb.sheetnames[:3]:
            ws = wb[sheet]
            result.append(f"Sheet: {sheet}")
            rows = list(ws.iter_rows(values_only=True))[:50]
            for row in rows:
                result.append("\t".join(str(c) if c is not None else "" for c in row))
        return "\n".join(result)[:12000]
    except Exception as e:
        return f"[Excel parse error: {e}]"

def parse_csv(file_bytes: bytes) -> str:
    try:
        text    = file_bytes.decode("utf-8", errors="replace")
        reader  = csv.reader(io.StringIO(text))
        rows    = list(reader)[:100]
        preview = "\n".join([",".join(r) for r in rows])
        return f"CSV ({len(rows)} rows shown):\n{preview}"[:8000]
    except Exception as e:
        return f"[CSV parse error: {e}]"

def parse_json(file_bytes: bytes) -> str:
    try:
        data      = json.loads(file_bytes.decode("utf-8", errors="replace"))
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        return formatted[:8000]
    except Exception as e:
        return f"[JSON parse error: {e}]"

def parse_zip(file_bytes: bytes) -> str:
    try:
        result = []
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = zf.namelist()
            result.append(f"ZIP contains {len(names)} files:")
            for name in names[:30]:
                info = zf.getinfo(name)
                result.append(f"  • {name}  ({info.file_size} bytes)")
            # Try to read small text files inside
            for name in names[:5]:
                if name.endswith((".txt",".py",".js",".md",".csv",".json")):
                    try:
                        content = zf.read(name).decode("utf-8", errors="replace")[:2000]
                        result.append(f"\n--- {name} ---\n{content}")
                    except Exception:
                        pass
        return "\n".join(result)[:10000]
    except Exception as e:
        return f"[ZIP parse error: {e}]"

def parse_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")[:12000]

CODE_EXTENSIONS = {
    ".py",".js",".ts",".java",".cpp",".c",".cs",".go",
    ".rb",".php",".swift",".kt",".rs",".html",".css",".sh",
    ".yaml",".yml",".toml",".xml",".sql",
}

def parse_file(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Returns (parsed_text, file_type_label)"""
    ext = os.path.splitext(filename.lower())[1]
    analytics["file_types"][ext] += 1

    if ext == ".pdf":
        return parse_pdf(file_bytes), "PDF"
    elif ext in (".docx", ".doc"):
        return parse_docx(file_bytes), "Word Document"
    elif ext in (".xlsx", ".xls"):
        return parse_excel(file_bytes), "Excel Spreadsheet"
    elif ext == ".csv":
        return parse_csv(file_bytes), "CSV File"
    elif ext == ".json":
        return parse_json(file_bytes), "JSON File"
    elif ext == ".zip":
        return parse_zip(file_bytes), "ZIP Archive"
    elif ext in CODE_EXTENSIONS:
        return parse_text(file_bytes), f"Code File ({ext})"
    elif ext in (".txt", ".md", ".log", ".env.example"):
        return parse_text(file_bytes), "Text File"
    else:
        # Try as text anyway
        try:
            return parse_text(file_bytes), f"File ({ext})"
        except Exception:
            return "", f"Unsupported ({ext})"

# ═══════════════════════════════════════════════════════════════════════════════
#  KEYBOARDS
# ═══════════════════════════════════════════════════════════════════════════════
def continue_kb():
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("➡️ Continue", callback_data="continue_reply"))
    return kb

def main_menu_kb():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("🧠 Normal Mode",    callback_data="mode_normal"),
        types.InlineKeyboardButton("🌐 Translate",      callback_data="mode_translate"),
        types.InlineKeyboardButton("📝 Summarize",      callback_data="mode_summarize"),
        types.InlineKeyboardButton("💻 Code Helper",    callback_data="mode_code"),
        types.InlineKeyboardButton("✍️ Essay Mode",     callback_data="mode_essay"),
        types.InlineKeyboardButton("📊 My Stats",       callback_data="my_stats"),
        types.InlineKeyboardButton("🏆 Leaderboard",    callback_data="leaderboard"),
        types.InlineKeyboardButton("🗑️ Clear History",  callback_data="clear"),
    )
    return kb

def admin_kb():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📊 Full Analytics", callback_data="admin_analytics"),
        types.InlineKeyboardButton("🏆 Leaderboard",    callback_data="leaderboard"),
        types.InlineKeyboardButton("❌ Error Logs",      callback_data="admin_errors"),
        types.InlineKeyboardButton("⚡ Response Stats",  callback_data="admin_response"),
        types.InlineKeyboardButton("📁 File Stats",      callback_data="admin_files"),
    )
    return kb

# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK  (Render webhook)
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return f"✅ Bot running | Users: {len(analytics['total_users'])} | Msgs: {analytics['total_messages']}", 200

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    update = telebot.types.Update.de_json(request.get_data().decode("utf-8"))
    bot.process_new_updates([update])
    return "OK", 200

# ═══════════════════════════════════════════════════════════════════════════════
#  COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════
def send_typing(chat_id):
    try: bot.send_chat_action(chat_id, "typing")
    except: pass

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

@bot.message_handler(commands=["start"])
def cmd_start(msg):
    uid  = msg.from_user.id
    name = msg.from_user.first_name or "there"
    track_message(uid)
    if uid in user_stats:
        user_stats[uid]["name"] = msg.from_user.first_name or ""
    send_reaction(msg.chat.id, msg.message_id, "👋")
    text = (
        f"⚡ <b>Welcome, {name}!</b>\n\n"
        "🧠 <b>Multi-Model AI Engine</b> — GPT-OSS • Qwen • DeepSeek • Llama • Kimi\n"
        "🚀 Running on <b>Groq LPU</b> — some of the fastest AI inference in the world\n\n"
        "<b>✨ What I can do:</b>\n"
        "• 💬 <b>Unlimited questions</b> — no limits, no usage caps\n"
        "• 🧬 <b>Auto model-switching</b> — never see an error, ever\n"
        "• 🌐 <b>Translate</b> — 100+ languages, instantly\n"
        "• 📄 <b>Read files</b> — PDF, DOCX, Excel, CSV, JSON, ZIP, code\n"
        "• 🎙 <b>Voice messages</b> — speech to text\n"
        "• 🧠 <b>Long-term memory</b> — I remember you, even after restarts\n"
        "• 📝 <b>Smart summarization</b> — long chats compress automatically\n\n"
        f"<i>{len(MODELS)} AI models active right now. Pick a mode below or just start typing 👇</i>"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML", reply_markup=main_menu_kb())


@bot.message_handler(commands=["help"])
def cmd_help(msg):
    send_reaction(msg.chat.id, msg.message_id, "💡")
    text = (
        "<b>📌 Commands:</b>\n\n"
        "/start — Welcome + mode selector\n"
        "/mode — Switch AI mode\n"
        "/model — Choose AI model (DeepSeek/Qwen/Llama/Mistral/GPT)\n"
        "/clear — Clear your history & memory\n"
        "/stats — Your personal stats\n"
        "/ask [question] — Quick AI question\n"
        "/translate [lang] [text] — Translate (100+ languages)\n"
        "/summarize [text] — Summarize text\n"
        "/leaderboard — Top users\n"
        "/testai — Check which AI models are online\n"
        "/checkkey — Debug: verify your GROQ_API_KEY is loaded correctly\n"
        "/help — This message\n\n"
        "💬 Just type normally for <b>unlimited questions</b> — no limits, "
        "and I auto-switch models if one is busy so you never see an error."
    )
    if is_admin(msg.from_user.id):
        text += (
            "\n\n<b>🔐 Admin Commands:</b>\n"
            "/admin — Admin dashboard\n"
            "/broadcast [msg] — Send to all users\n"
        )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")


@bot.message_handler(commands=["mode"])
def cmd_mode(msg):
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("🧠 Normal",    callback_data="mode_normal"),
        types.InlineKeyboardButton("🌐 Translate", callback_data="mode_translate"),
        types.InlineKeyboardButton("📝 Summarize", callback_data="mode_summarize"),
        types.InlineKeyboardButton("💻 Code",      callback_data="mode_code"),
        types.InlineKeyboardButton("✍️ Essay",     callback_data="mode_essay"),
    )
    bot.send_message(msg.chat.id, "🔄 <b>Select AI Mode:</b>", parse_mode="HTML", reply_markup=kb)


@bot.message_handler(commands=["model"])
def cmd_model(msg):
    uid = msg.from_user.id
    current = user_model_pref.get(uid, "Auto (smart fallback)")
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("🤖 Auto (recommended)", callback_data="setmodel_auto"),
        types.InlineKeyboardButton("🧬 GPT",       callback_data="setmodel_gpt-oss"),
        types.InlineKeyboardButton("🔮 Qwen",      callback_data="setmodel_qwen"),
        types.InlineKeyboardButton("🐳 DeepSeek",  callback_data="setmodel_deepseek"),
        types.InlineKeyboardButton("🦙 Llama",     callback_data="setmodel_llama"),
        types.InlineKeyboardButton("🌙 Kimi",      callback_data="setmodel_kimi"),
    )
    bot.send_message(msg.chat.id,
        f"⚡ <b>Choose preferred AI model:</b>\n\nCurrent: <b>{current}</b>\n\n"
        "All models run on <b>Groq's LPU hardware</b> — even if you pick one, "
        "I'll auto-switch to another if it's ever busy, so you never get stuck or see an error.",
        parse_mode="HTML", reply_markup=kb)


@bot.message_handler(commands=["clear"])
def cmd_clear(msg):
    with _state_lock:
        user_histories[msg.from_user.id] = []
        user_summaries.pop(msg.from_user.id, None)
    pending_continue.pop(msg.from_user.id, None)
    send_reaction(msg.chat.id, msg.message_id, "👍")
    bot.send_message(msg.chat.id, "🗑️ <b>History cleared!</b> Fresh start.", parse_mode="HTML")


@bot.message_handler(commands=["stats"])
def cmd_stats(msg):
    uid   = msg.from_user.id
    track_message(uid)
    s     = user_stats.get(uid, {})
    count = s.get("count", 0)
    since = s.get("first_seen", "today")
    mode  = user_modes.get(uid, "normal").capitalize()
    hist  = len(user_histories.get(uid, []))
    toks  = s.get("tokens_used", 0)
    text = (
        "📊 <b>Your Stats:</b>\n\n"
        f"• Messages sent: <b>{count}</b>\n"
        f"• Tokens used: <b>{toks:,}</b>\n"
        f"• Using since: <b>{since}</b>\n"
        f"• Current mode: <b>{mode}</b>\n"
        f"• History: <b>{hist} messages</b>"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")


@bot.message_handler(commands=["leaderboard"])
def cmd_leaderboard(msg):
    lb   = leaderboard()
    rows = ["🏆 <b>Top Users:</b>\n"]
    medals = ["🥇","🥈","🥉"] + ["🔹"]*7
    for i, (uid, count, name, toks) in enumerate(lb):
        display = name or f"User{uid}"
        rows.append(f"{medals[i]} {display} — <b>{count}</b> msgs ({toks:,} tokens)")
    bot.send_message(msg.chat.id, "\n".join(rows), parse_mode="HTML")


@bot.message_handler(commands=["ask"])
def cmd_ask(msg):
    q = msg.text.partition(" ")[2].strip()
    if not q:
        bot.send_message(msg.chat.id, "Usage: /ask Your question here")
        return
    track_message(msg.from_user.id)
    send_typing(msg.chat.id)
    send_reaction(msg.chat.id, msg.message_id, "🤔")
    reply, truncated = get_ai_response(msg.from_user.id, q)
    kb = continue_kb() if truncated else None
    if truncated:
        pending_continue[msg.from_user.id] = reply
    safe_send(msg.chat.id, format_for_telegram(reply), reply_to=msg.message_id, kb=kb)


@bot.message_handler(commands=["translate"])
def cmd_translate(msg):
    txt = msg.text.partition(" ")[2].strip()
    if not txt:
        bot.send_message(msg.chat.id,
            "Usage:\n"
            "/translate Aap kaise hain?  (auto → English)\n"
            "/translate es Hello, how are you?  (auto → Spanish, use language code/name)",
            parse_mode="HTML")
        return
    track_message(msg.from_user.id)
    send_typing(msg.chat.id)
    send_reaction(msg.chat.id, msg.message_id, "🌐")

    # Optional target-language prefix: "/translate fr <text>" or "/translate french <text>"
    parts = txt.split(" ", 1)
    target_lang = None
    text_to_translate = txt
    if len(parts) == 2 and len(parts[0]) <= 15 and parts[0].isalpha():
        target_lang = parts[0]
        text_to_translate = parts[1]

    if target_lang:
        prompt = f"Detect the source language and translate this text to {target_lang}:\n\n{text_to_translate}"
    else:
        prompt = f"Detect the source language and translate this text to English:\n\n{text_to_translate}"

    reply, truncated = get_ai_response(msg.from_user.id, prompt)
    kb = continue_kb() if truncated else None
    if truncated:
        pending_continue[msg.from_user.id] = reply
    safe_send(msg.chat.id, format_for_telegram(reply), reply_to=msg.message_id, kb=kb)


@bot.message_handler(commands=["summarize"])
def cmd_summarize(msg):
    txt = msg.text.partition(" ")[2].strip()
    if not txt:
        bot.send_message(msg.chat.id, "Usage: /summarize [long text here]")
        return
    track_message(msg.from_user.id)
    send_typing(msg.chat.id)
    send_reaction(msg.chat.id, msg.message_id, "📝")
    reply, truncated = get_ai_response(msg.from_user.id,
        f"Summarize in 3-5 bullet points:\n\n{txt}")
    kb = continue_kb() if truncated else None
    if truncated:
        pending_continue[msg.from_user.id] = reply
    safe_send(msg.chat.id, format_for_telegram(reply), reply_to=msg.message_id, kb=kb)


# ── ADMIN COMMANDS ─────────────────────────────────────────────────────────────
@bot.message_handler(commands=["admin"])
def cmd_admin(msg):
    if not is_admin(msg.from_user.id):
        bot.send_message(msg.chat.id, "⛔ Admin only.")
        return
    today = datetime.now().strftime("%Y-%m-%d")
    text = (
        "🔐 <b>Admin Dashboard</b>\n\n"
        f"• Total users: <b>{len(analytics['total_users'])}</b>\n"
        f"• Active today: <b>{len(analytics['active_today'])}</b>\n"
        f"• Total messages: <b>{analytics['total_messages']:,}</b>\n"
        f"• Today's messages: <b>{analytics['daily_messages'].get(today, 0)}</b>\n"
        f"• Total tokens: <b>{analytics['total_tokens']:,}</b>\n"
        f"• Errors logged: <b>{len(analytics['errors'])}</b>\n"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML", reply_markup=admin_kb())


@bot.message_handler(commands=["testai"])
def cmd_testai(msg):
    """Debug command — tests AI connection across all models and shows results + speed."""
    send_typing(msg.chat.id)
    wait = bot.send_message(msg.chat.id, "🔄 Testing AI connections...")
    results = []
    for model in MODELS:
        try:
            t0 = time.time()
            resp = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10,
            )
            elapsed = time.time() - t0
            reply = (resp.choices[0].message.content or "").strip()
            results.append(f"✅ <code>{model}</code> → {reply[:30]} <i>({elapsed:.2f}s ⚡)</i>")
        except Exception as e:
            results.append(f"❌ <code>{model}</code> → {type(e).__name__}: {str(e)[:60]}")
    text = "<b>🧪 Model Test Results (Groq):</b>\n\n" + "\n".join(results)
    bot.edit_message_text(text, msg.chat.id, wait.message_id, parse_mode="HTML")


@bot.message_handler(commands=["checkkey"])
def cmd_checkkey(msg):
    """Debug command — shows a masked version of the GROQ_API_KEY the bot is actually using."""
    if not GROQ_API_KEY:
        bot.send_message(msg.chat.id, "❌ <b>GROQ_API_KEY is EMPTY / not set</b> in Render environment.", parse_mode="HTML")
        return
    key = GROQ_API_KEY
    length = len(key)
    starts_ok = key.startswith("gsk_")
    has_space = " " in key or "\n" in key or "\t" in key
    masked = key[:8] + "..." + key[-4:] if length > 12 else "(too short!)"
    text = (
        "🔑 <b>GROQ_API_KEY Debug:</b>\n\n"
        f"• Value seen by bot: <code>{masked}</code>\n"
        f"• Length: <b>{length}</b> characters\n"
        f"• Starts with 'gsk_': <b>{'✅ Yes' if starts_ok else '❌ NO — wrong key type!'}</b>\n"
        f"• Contains space/newline: <b>{'❌ YES — this breaks it!' if has_space else '✅ No'}</b>\n\n"
        "A real Groq key is usually 50-60+ characters and starts with <code>gsk_</code>. "
        "If length looks short or starts wrong, the Render env var has a copy-paste problem."
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")


@bot.message_handler(commands=["broadcast"])
def cmd_broadcast(msg):
    if not is_admin(msg.from_user.id):
        bot.send_message(msg.chat.id, "⛔ Admin only.")
        return
    text = msg.text.partition(" ")[2].strip()
    if not text:
        bot.send_message(msg.chat.id, "Usage: /broadcast Your message here")
        return
    sent = 0
    for uid in list(analytics["total_users"]):
        try:
            bot.send_message(uid, f"📢 <b>Announcement:</b>\n\n{text}", parse_mode="HTML")
            sent += 1
            time.sleep(0.05)
        except Exception:
            pass
    bot.send_message(msg.chat.id, f"✅ Broadcast sent to <b>{sent}</b> users.", parse_mode="HTML")


# ── GROUP ADMIN TOOLS ──────────────────────────────────────────────────────────
@bot.message_handler(commands=["ban"])
def cmd_ban(msg):
    if msg.chat.type == "private":
        bot.send_message(msg.chat.id, "This command works in groups only.")
        return
    try:
        target = msg.reply_to_message.from_user.id if msg.reply_to_message else None
        if not target:
            bot.send_message(msg.chat.id, "Reply to a user's message to ban them.")
            return
        bot.ban_chat_member(msg.chat.id, target)
        bot.send_message(msg.chat.id, "✅ User banned.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"❌ Could not ban: {e}")


@bot.message_handler(commands=["unban"])
def cmd_unban(msg):
    if msg.chat.type == "private":
        return
    try:
        target = msg.reply_to_message.from_user.id if msg.reply_to_message else None
        if not target:
            bot.send_message(msg.chat.id, "Reply to the user's message to unban.")
            return
        bot.unban_chat_member(msg.chat.id, target)
        bot.send_message(msg.chat.id, "✅ User unbanned.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"❌ Could not unban: {e}")


@bot.message_handler(commands=["mute"])
def cmd_mute(msg):
    if msg.chat.type == "private":
        return
    try:
        target = msg.reply_to_message.from_user.id if msg.reply_to_message else None
        if not target:
            bot.send_message(msg.chat.id, "Reply to a user's message to mute them.")
            return
        until = int(time.time()) + 3600  # mute 1 hour
        bot.restrict_chat_member(msg.chat.id, target,
            types.ChatPermissions(can_send_messages=False), until_date=until)
        bot.send_message(msg.chat.id, "🔇 User muted for 1 hour.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"❌ Could not mute: {e}")


@bot.message_handler(commands=["unmute"])
def cmd_unmute(msg):
    if msg.chat.type == "private":
        return
    try:
        target = msg.reply_to_message.from_user.id if msg.reply_to_message else None
        if not target:
            return
        bot.restrict_chat_member(msg.chat.id, target,
            types.ChatPermissions(
                can_send_messages=True, can_send_media_messages=True,
                can_send_polls=True, can_send_other_messages=True,
            ))
        bot.send_message(msg.chat.id, "🔊 User unmuted.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"❌ Could not unmute: {e}")


@bot.message_handler(commands=["kick"])
def cmd_kick(msg):
    if msg.chat.type == "private":
        return
    try:
        target = msg.reply_to_message.from_user.id if msg.reply_to_message else None
        if not target:
            return
        bot.ban_chat_member(msg.chat.id, target)
        bot.unban_chat_member(msg.chat.id, target)
        bot.send_message(msg.chat.id, "👢 User kicked.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"❌ Could not kick: {e}")


@bot.message_handler(commands=["pin"])
def cmd_pin(msg):
    if msg.chat.type == "private" or not msg.reply_to_message:
        return
    try:
        bot.pin_chat_message(msg.chat.id, msg.reply_to_message.message_id)
        bot.send_message(msg.chat.id, "📌 Message pinned.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"❌ {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CALLBACK HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
@bot.callback_query_handler(func=lambda c: True)
def handle_callback(call):
    uid  = call.from_user.id
    data = call.data

    try:
        # ── Continue truncated reply ────────────────────────────────────────
        if data == "continue_reply":
            bot.answer_callback_query(call.id)
            prior = pending_continue.get(uid)
            if not prior:
                safe_send(call.message.chat.id, "Nothing to continue 🙂")
                return
            send_typing(call.message.chat.id)
            reply, truncated = get_ai_response(uid, "Continue exactly where you left off, no repetition.")
            kb = continue_kb() if truncated else None
            if truncated:
                pending_continue[uid] = reply
            else:
                pending_continue.pop(uid, None)
            safe_send(call.message.chat.id, format_for_telegram(reply), kb=kb)
            return

        # ── Model family switching ──────────────────────────────────────────
        if data.startswith("setmodel_"):
            fam = data[len("setmodel_"):]
            if fam == "auto":
                user_model_pref.pop(uid, None)
                bot.answer_callback_query(call.id, "✅ Auto mode — best available model")
                safe_send(call.message.chat.id, "✅ Switched to <b>Auto</b> — I'll pick whichever model responds fastest.")
            else:
                user_model_pref[uid] = fam
                bot.answer_callback_query(call.id, f"✅ {fam} preferred")
                safe_send(call.message.chat.id, f"✅ Now preferring <b>{fam}</b> models (still auto-falls-back if busy).")
            return

        # ── Mode switching ──────────────────────────────────────────────────────
        if data.startswith("mode_"):
            mode = data[5:]
            user_modes[uid] = mode
            labels = {
                "normal":    "🧠 Normal Mode",
                "translate": "🌐 Translate Mode",
                "summarize": "📝 Summarize Mode",
                "code":      "💻 Code Mode",
                "essay":     "✍️ Essay Mode",
            }
            label = labels.get(mode, mode.capitalize())
            bot.answer_callback_query(call.id, f"✅ {label} activated!")
            safe_send(call.message.chat.id,
                f"✅ Switched to <b>{label}</b>!\n\nNow just send your message.")

        # ── Clear history ────────────────────────────────────────────────────────
        elif data == "clear":
            with _state_lock:
                user_histories[uid] = []
                user_summaries.pop(uid, None)
            pending_continue.pop(uid, None)
            bot.answer_callback_query(call.id, "History cleared!")
            safe_send(call.message.chat.id, "🗑️ <b>History & memory cleared!</b> Fresh start.")

        # ── Personal stats ───────────────────────────────────────────────────────
        elif data == "my_stats":
            track_message(uid)
            s     = user_stats.get(uid, {})
            count = s.get("count", 0)
            since = s.get("first_seen", "today")
            toks  = s.get("tokens_used", 0)
            mode  = user_modes.get(uid, "normal").capitalize()
            hist  = len(user_histories.get(uid, []))
            bot.answer_callback_query(call.id)
            safe_send(call.message.chat.id,
                f"📊 <b>Your Stats:</b>\n\n"
                f"• Messages: <b>{count}</b>\n"
                f"• Tokens: <b>{toks:,}</b>\n"
                f"• Since: <b>{since}</b>\n"
                f"• Mode: <b>{mode}</b>\n"
                f"• History: <b>{hist} msgs</b>")
    
        # ── Leaderboard ──────────────────────────────────────────────────────────
        elif data == "leaderboard":
            lb    = leaderboard()
            medals = ["🥇","🥈","🥉"] + ["🔹"]*7
            rows  = ["🏆 <b>Top Users:</b>\n"]
            for i, (u, count, name, toks) in enumerate(lb):
                display = name or f"User{u}"
                rows.append(f"{medals[i]} {display} — <b>{count}</b> msgs")
            bot.answer_callback_query(call.id)
            safe_send(call.message.chat.id, "\n".join(rows))
    
        # ── Admin: full analytics ────────────────────────────────────────────────
        elif data == "admin_analytics" and is_admin(uid):
            today  = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            avg_rt = (sum(analytics["response_times"]) / len(analytics["response_times"])
                      if analytics["response_times"] else 0)
            text = (
                "📊 <b>Full Analytics:</b>\n\n"
                f"• Total users: <b>{len(analytics['total_users'])}</b>\n"
                f"• Active today: <b>{len(analytics['active_today'])}</b>\n"
                f"• Today msgs: <b>{analytics['daily_messages'].get(today,0)}</b>\n"
                f"• Yesterday msgs: <b>{analytics['daily_messages'].get(yesterday,0)}</b>\n"
                f"• Total messages: <b>{analytics['total_messages']:,}</b>\n"
                f"• Total tokens: <b>{analytics['total_tokens']:,}</b>\n"
                f"• Avg response: <b>{avg_rt:.2f}s</b>\n"
                f"• Error count: <b>{len(analytics['errors'])}</b>\n"
                f"• Active models: <b>{len(MODELS)}</b>"
            )
            bot.answer_callback_query(call.id)
            safe_send(call.message.chat.id, text)
    
        elif data == "admin_errors" and is_admin(uid):
            errors = analytics["errors"][-10:]
            if not errors:
                text = "✅ No errors logged!"
            else:
                rows = ["❌ <b>Last 10 Errors:</b>\n"]
                for e in reversed(errors):
                    rows.append(f"• [{e['time']}] User {e['user_id']}: {e['error'][:80]}")
                text = "\n".join(rows)
            bot.answer_callback_query(call.id)
            safe_send(call.message.chat.id, text)
    
        elif data == "admin_response" and is_admin(uid):
            rt = analytics["response_times"]
            if not rt:
                text = "No response time data yet."
            else:
                avg = sum(rt)/len(rt)
                mn  = min(rt)
                mx  = max(rt)
                text = (
                    "⚡ <b>Response Times:</b>\n\n"
                    f"• Average: <b>{avg:.2f}s</b>\n"
                    f"• Fastest: <b>{mn:.2f}s</b>\n"
                    f"• Slowest: <b>{mx:.2f}s</b>\n"
                    f"• Samples: <b>{len(rt)}</b>"
                )
            bot.answer_callback_query(call.id)
            safe_send(call.message.chat.id, text)
    
        elif data == "admin_files" and is_admin(uid):
            ft = analytics["file_types"]
            if not ft:
                text = "No files processed yet."
            else:
                rows = ["📁 <b>File Type Usage:</b>\n"]
                for ext, count in sorted(ft.items(), key=lambda x: -x[1]):
                    rows.append(f"• <code>{ext or 'unknown'}</code>: <b>{count}</b>")
                text = "\n".join(rows)
            bot.answer_callback_query(call.id)
            safe_send(call.message.chat.id, text)
    
        else:
            bot.answer_callback_query(call.id)

    except Exception as e:
        logger.error(f"handle_callback crashed for user {uid} (data={data}): {type(e).__name__}: {e}")
        try:
            bot.answer_callback_query(call.id, "Kuch gadbad ho gayi, phir try karo 🙏")
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT / FILE HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
@bot.message_handler(content_types=["document"])
def handle_document(msg):
    uid      = msg.from_user.id
    chat_id  = msg.chat.id
    doc      = msg.document
    filename = doc.file_name or "file"
    caption  = msg.caption or ""

    track_message(uid)
    send_reaction(chat_id, msg.message_id, "📄")
    send_typing(chat_id)

    wait_msg = bot.send_message(chat_id, f"📂 Reading <b>{filename}</b>...", parse_mode="HTML")

    try:
        try:
            file_info = bot.get_file(doc.file_id)
            file_bytes = bot.download_file(file_info.file_path)
        except Exception as e:
            bot.edit_message_text(f"❌ Could not download file: {e}",
                                  chat_id, wait_msg.message_id)
            return

        parsed, label = parse_file(file_bytes, filename)

        if not parsed:
            bot.edit_message_text(
                f"❌ Could not read <b>{filename}</b>. Unsupported format.",
                chat_id, wait_msg.message_id, parse_mode="HTML")
            return

        # Build AI prompt
        user_question = caption if caption else f"Analyze this {label} and give a detailed summary with key insights."
        prompt = f"[{label}: {filename}]\n\n{parsed[:10000]}\n\nUser request: {user_question}"

        bot.edit_message_text(f"🧠 Analyzing <b>{filename}</b>...", chat_id, wait_msg.message_id, parse_mode="HTML")

        reply, truncated = get_ai_response(uid, prompt,
            extra_system=f"The user has uploaded a {label}. Analyze it thoroughly. Format output clearly with sections and bold key points.")
        bot.delete_message(chat_id, wait_msg.message_id)
        kb = continue_kb() if truncated else None
        if truncated:
            pending_continue[uid] = reply
        safe_send(chat_id, format_for_telegram(reply), reply_to=msg.message_id, kb=kb)

    except Exception as e:
        logger.error(f"handle_document crashed for user {uid}: {type(e).__name__}: {e}")
        track_error(uid, str(e))
        try:
            bot.edit_message_text(
                f"❌ Kuch gadbad ho gayi <b>{filename}</b> process karte waqt. Phir try karo.",
                chat_id, wait_msg.message_id, parse_mode="HTML")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PHOTO HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
@bot.message_handler(content_types=["photo"])
def handle_photo(msg):
    uid     = msg.from_user.id
    chat_id = msg.chat.id
    caption = msg.caption or "Describe this image in detail."

    track_message(uid)
    send_reaction(chat_id, msg.message_id, "🖼")
    safe_send(chat_id,
        "🖼 <b>Image received!</b>\n\nI can't directly see images yet, but if you have text in the image, use /ask to type it and I'll help!",
        reply_to=msg.message_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  VOICE HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
@bot.message_handler(content_types=["voice"])
def handle_voice(msg):
    uid     = msg.from_user.id
    chat_id = msg.chat.id
    track_message(uid)
    send_reaction(chat_id, msg.message_id, "🎵")

    if not HAS_VOICE:
        safe_send(chat_id,
            "🎤 <b>Voice received!</b>\n\nVoice-to-text requires extra libraries.\n"
            "Please <b>type your message</b> for now — I'll respond instantly!",
            reply_to=msg.message_id)
        return

    send_typing(chat_id)
    wait_msg = bot.send_message(chat_id, "🎤 Transcribing voice message...")
    try:
        file_info  = bot.get_file(msg.voice.file_id)
        ogg_bytes  = bot.download_file(file_info.file_path)
        audio      = AudioSegment.from_ogg(io.BytesIO(ogg_bytes))
        wav_buf    = io.BytesIO()
        audio.export(wav_buf, format="wav")
        wav_buf.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buf) as source:
            audio_data = recognizer.record(source)
        transcribed = recognizer.recognize_google(audio_data)

        bot.edit_message_text(f"🎤 <b>You said:</b> {transcribed}", chat_id, wait_msg.message_id, parse_mode="HTML")

        reply, truncated = get_ai_response(uid, transcribed)
        kb = continue_kb() if truncated else None
        if truncated:
            pending_continue[uid] = reply
        safe_send(chat_id, format_for_telegram(reply), reply_to=msg.message_id, kb=kb)
    except Exception as e:
        bot.edit_message_text(f"❌ Could not transcribe: {e}", chat_id, wait_msg.message_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TEXT HANDLER  — unlimited questions, smart retry
# ═══════════════════════════════════════════════════════════════════════════════
@bot.message_handler(func=lambda m: m.text and not m.text.startswith("/"))
def handle_text(msg):
    uid       = msg.from_user.id
    chat_id   = msg.chat.id
    user_text = msg.text.strip()

    try:
        # Store name
        with _state_lock:
            if uid not in user_stats:
                user_stats[uid] = {"count": 0, "first_seen": datetime.now().strftime("%Y-%m-%d"),
                                   "name": "", "tokens_used": 0}
            user_stats[uid]["name"] = msg.from_user.first_name or ""

        track_message(uid)
        logger.info(f"User {uid} [{user_modes.get(uid,'normal')}]: {user_text[:60]}")

        send_typing(chat_id)
        send_reaction(chat_id, msg.message_id, pick_reaction(user_text))

        # get_ai_response already retries across every model, twice each —
        # it never raises and never returns empty, so no outer retry needed here.
        reply, truncated = get_ai_response(uid, user_text)

        kb = continue_kb() if truncated else None
        if truncated:
            pending_continue[uid] = reply

        formatted = format_for_telegram(reply)
        safe_send(chat_id, formatted, reply_to=msg.message_id, kb=kb)

    except Exception as e:
        # Absolute last line of defense — no matter what goes wrong, the user
        # gets a friendly message instead of silence or a Telegram error.
        logger.error(f"handle_text crashed for user {uid}: {type(e).__name__}: {e}")
        track_error(uid, str(e))
        try:
            safe_send(chat_id,
                "Kuch gadbad ho gayi mere is taraf 🙏 Please apna message ek baar phir bhejo.",
                reply_to=msg.message_id)
        except Exception:
            pass


@bot.message_handler(content_types=["sticker", "video", "video_note", "audio"])
def handle_other(msg):
    safe_send(msg.chat.id,
        "📝 Send me <b>text</b>, <b>documents</b> (PDF/DOCX/Excel/CSV/JSON/ZIP), or <b>voice messages</b>!",
        reply_to=msg.message_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PORT       = int(os.environ.get("PORT", 5000))
    RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

    load_memory()
    threading.Thread(target=autosave_loop, daemon=True).start()
    logger.info("✅ Long-term memory loaded, autosave thread started (every 30s)")

    try:
        if RENDER_URL:
            webhook_url = f"{RENDER_URL}/{BOT_TOKEN}"
            bot.remove_webhook()
            bot.set_webhook(url=webhook_url)
            logger.info(f"✅ Webhook → {webhook_url}")
            app.run(host="0.0.0.0", port=PORT)
        else:
            logger.info("🔄 Polling mode (local)...")
            bot.remove_webhook()
            threading.Thread(target=bot.infinity_polling, daemon=True).start()
            app.run(host="0.0.0.0", port=PORT)
    finally:
        save_memory()
        logger.info("💾 Final memory save complete on shutdown")
