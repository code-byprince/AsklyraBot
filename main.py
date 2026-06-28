# -*- coding: utf-8 -*-
# ================================================================
#  ADVANCED TELEGRAM AI BOT  —  Bynara API
#  All features included — Render.com ready
# ================================================================

import os, re, io, csv, json, time, random, logging, zipfile, threading, sqlite3
import requests as http_requests
from datetime import datetime, timedelta
from collections import defaultdict

from flask import Flask, request
import telebot
from telebot import types
from openai import OpenAI

# ── Optional libraries ───────────────────────────────────────────
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

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Environment Variables ────────────────────────────────────────
BOT_TOKEN  = os.environ.get("BOT_TOKEN", "")
NARA_TOKEN = os.environ.get("NARA_TOKEN", "")
ADMIN_IDS  = [int(x) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip().isdigit()]

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not set!")
if not NARA_TOKEN:
    raise RuntimeError("NARA_TOKEN not set!")

# ── Clients ──────────────────────────────────────────────────────
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)

ai_client = OpenAI(
    base_url="https://router.bynara.id/v1",
    api_key=NARA_TOKEN,
)

MODEL      = "auto/bynara"
MAX_TOKENS = 1500

# ================================================================
#  DATABASE  — SQLite for persistent memory
# ================================================================
DB_PATH = "/tmp/bot_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     INTEGER PRIMARY KEY,
            name        TEXT,
            first_seen  TEXT,
            msg_count   INTEGER DEFAULT 0,
            tokens_used INTEGER DEFAULT 0,
            language    TEXT DEFAULT 'auto',
            mode        TEXT DEFAULT 'normal'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER,
            role       TEXT,
            content    TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            user_id    INTEGER PRIMARY KEY,
            summary    TEXT,
            updated_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER,
            error      TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def db():
    return sqlite3.connect(DB_PATH)

# ── User DB helpers ──────────────────────────────────────────────
def ensure_user(user_id, name=""):
    with db() as conn:
        today = datetime.now().strftime("%Y-%m-%d")
        conn.execute("""
            INSERT INTO users (user_id, name, first_seen, msg_count, tokens_used)
            VALUES (?, ?, ?, 0, 0)
            ON CONFLICT(user_id) DO UPDATE SET name=excluded.name
        """, (user_id, name, today))
        conn.commit()

def get_user(user_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        return None
    cols = ["user_id","name","first_seen","msg_count","tokens_used","language","mode"]
    return dict(zip(cols, row))

def update_user_stat(user_id, msg_count_delta=0, tokens_delta=0):
    with db() as conn:
        conn.execute("""
            UPDATE users SET
                msg_count   = msg_count   + ?,
                tokens_used = tokens_used + ?
            WHERE user_id = ?
        """, (msg_count_delta, tokens_delta, user_id))
        conn.commit()

def set_user_mode(user_id, mode):
    with db() as conn:
        conn.execute("UPDATE users SET mode=? WHERE user_id=?", (mode, user_id))
        conn.commit()

def set_user_language(user_id, lang):
    with db() as conn:
        conn.execute("UPDATE users SET language=? WHERE user_id=?", (lang, user_id))
        conn.commit()

# ── Message history (persistent) ────────────────────────────────
HISTORY_LIMIT    = 40   # messages kept in DB per user
SUMMARY_AFTER    = 30   # summarize when history exceeds this

def save_message(user_id, role, content):
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?,?,?,?)",
            (user_id, role, content[:4000], datetime.now().isoformat())
        )
        conn.commit()
        # Trim old messages
        conn.execute("""
            DELETE FROM messages WHERE id IN (
                SELECT id FROM messages WHERE user_id=?
                ORDER BY id DESC LIMIT -1 OFFSET ?
            )
        """, (user_id, HISTORY_LIMIT))
        conn.commit()

def get_history(user_id):
    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE user_id=? ORDER BY id ASC",
            (user_id,)
        ).fetchall()
    return [{"role": r, "content": c} for r, c in rows]

def clear_history(user_id):
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM summaries WHERE user_id=?", (user_id,))
        conn.commit()

def save_summary(user_id, summary):
    with db() as conn:
        conn.execute("""
            INSERT INTO summaries (user_id, summary, updated_at) VALUES (?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at
        """, (user_id, summary, datetime.now().isoformat()))
        conn.commit()

def get_summary(user_id):
    with db() as conn:
        row = conn.execute("SELECT summary FROM summaries WHERE user_id=?", (user_id,)).fetchone()
    return row[0] if row else None

# ── Analytics DB ─────────────────────────────────────────────────
_analytics_cache = {
    "total_messages": 0,
    "total_tokens":   0,
    "response_times": [],
    "daily_messages": defaultdict(int),
    "active_today":   set(),
    "total_users":    set(),
    "model_calls":    0,
    "errors_count":   0,
}

def track_analytics(user_id, tokens=0, response_time=0.0):
    today = datetime.now().strftime("%Y-%m-%d")
    _analytics_cache["total_messages"]    += 1
    _analytics_cache["total_tokens"]      += tokens
    _analytics_cache["model_calls"]       += 1
    _analytics_cache["daily_messages"][today] += 1
    _analytics_cache["active_today"].add(user_id)
    _analytics_cache["total_users"].add(user_id)
    if response_time:
        _analytics_cache["response_times"].append(round(response_time, 3))
        _analytics_cache["response_times"] = _analytics_cache["response_times"][-500:]

def log_error(user_id, error):
    _analytics_cache["errors_count"] += 1
    with db() as conn:
        conn.execute(
            "INSERT INTO errors (user_id, error, created_at) VALUES (?,?,?)",
            (user_id, str(error)[:300], datetime.now().isoformat())
        )
        conn.commit()

def get_errors(limit=10):
    with db() as conn:
        rows = conn.execute(
            "SELECT user_id, error, created_at FROM errors ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return rows

def get_leaderboard(limit=10):
    with db() as conn:
        rows = conn.execute(
            "SELECT user_id, name, msg_count, tokens_used FROM users ORDER BY msg_count DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return rows

def get_all_users():
    with db() as conn:
        rows = conn.execute("SELECT user_id FROM users").fetchall()
    return [r[0] for r in rows]

# ================================================================
#  SYSTEM PROMPTS
# ================================================================
BASE_SYSTEM = """You are an advanced, helpful AI assistant inside a Telegram bot.

STRICT FORMATTING:
- NEVER use # ## ### headings
- Bold key words: <b>word</b>
- Lists: start lines with -
- Code: <code>code</code>
- Be concise, clear, and friendly
- Always respond in the same language the user writes in
"""

MODE_PROMPTS = {
    "normal":    "",
    "translate": "Translate every message to English. Always identify the source language at the start.",
    "summarize": "Summarize every message in 3-5 bullet points maximum. Be very concise.",
    "code":      "You are a senior developer. Analyze code, fix bugs, explain clearly. Format code with <code> tags.",
    "essay":     "Help write, improve, and proofread essays. Give structured, clear feedback.",
    "creative":  "You are a creative writing assistant. Help with stories, poems, scripts, and creative content.",
}

SUMMARY_PROMPT = """Summarize this conversation in 3-5 sentences capturing the main topics, 
key information shared, and important context. Be brief but complete:

{history}"""

# ================================================================
#  REACTIONS  — Telegram-verified emoji list only
# ================================================================

# These are the ONLY emojis Telegram officially supports as reactions
# Using actual unicode characters (not escape codes) for max compatibility
TELEGRAM_REACTIONS = [
    "👍",  # thumbs up
    "👎",  # thumbs down
    "❤",      # red heart
    "🔥",  # fire
    "🥺",  # pleading face
    "💯",  # 100
    "👏",  # clapping
    "🤔",  # thinking
    "😂",  # laughing
    "😮",  # open mouth
    "🤨",  # raised eyebrow
    "👌",  # ok hand
    "🖕",  # middle finger (not used but valid)
    "💩",  # poop
    "🙏",  # folded hands / pray
    "🤡",  # clown
    "💥",  # collision / exploding head
    "💨",  # dashing away
    "🙌",  # raising hands
    "👀",  # eyes
    "💦",  # droplets
    "🎊",  # confetti ball
    "💫",  # dizzy
    "💤",  # zzz
    "😈",  # smiling devil
    "📈",  # chart up
    "📉",  # chart down
    "✅",      # check mark
    "❌",      # cross mark
    "❗",      # exclamation
    "❓",      # question mark
    "💡",  # light bulb
    "📢",  # loudspeaker
    "🏆",  # trophy
    "📄",  # document
    "🎵",  # music note
    "💰",  # money bag
    "💣",  # bomb
    "🤝",  # handshake
    "📳",  # vibration mode
]

# Smart reaction rules — keyword → emoji
REACTION_RULES = [
    # Greetings
    (["hello","hi","hey","namaste","salam","hola","bonjour","salut","ciao",
      "assalam","adaab","good morning","good evening","good night",
      "subah","shaam","raat"],
     "👍"),

    # Thanks / Gratitude
    (["thank","thanks","shukriya","dhanyawad","ty","tysm","grateful",
      "appreciate","meherbani","shukar"],
     "🙏"),

    # Love / Positive emotion
    (["love","pyar","cute","beautiful","amazing","awesome","wonderful",
      "fantastic","excellent","great","best","superb","brilliant"],
     "❤"),

    # Funny / Jokes
    (["funny","lol","haha","hehe","joke","maza","mazak","comedy",
      "laughing","hilarious","rofl","lmao"],
     "😂"),

    # Sad / Emotional
    (["sad","dukh","cry","upset","depressed","bura","hurt","pain",
      "heartbreak","lonely","missing","yaad"],
     "🥺"),

    # Wow / Surprise
    (["wow","incredible","unbelievable","shocking","omg","surprised",
      "whoa","OMG","shocking","mindblowing"],
     "😮"),

    # Code / Tech
    (["code","coding","python","javascript","java","script","function",
      "debug","program","developer","git","api","server","database",
      "html","css","bug","error","exception","stack"],
     "💥"),

    # File / Document
    (["pdf","docx","file","document","upload","excel","csv","zip",
      "read","extract","analyze","spreadsheet"],
     "📄"),

    # Music
    (["music","song","gaana","listen","playlist","singer","album",
      "guitar","piano","beats","rhythm"],
     "🎵"),

    # Money / Finance
    (["money","paisa","rupee","dollar","payment","price","cost",
      "salary","invest","profit","loss","bank","finance"],
     "💰"),

    # Ideas / Creative
    (["idea","suggest","plan","concept","creative","think","thought",
      "solution","strategy","invent","innovate"],
     "💡"),

    # Yes / Correct
    (["yes","correct","right","haan","bilkul","sure","absolutely",
      "exactly","perfect","agreed","confirmed","done","sahi"],
     "✅"),

    # No / Wrong
    (["no","nahi","galat","wrong","incorrect","nope","never","false",
      "disagree","denied","rejected"],
     "❌"),

    # News / Updates
    (["news","update","latest","breaking","trending","report","announcement",
      "launched","released","new"],
     "📢"),

    # Sports / Games
    (["cricket","football","sports","game","match","score","win","lose",
      "team","player","tournament","championship"],
     "🏆"),

    # Questions / Confusion
    (["kya","what","why","how","when","where","who","confused",
      "samajh","bata","explain","matlab","means"],
     "🤔"),

    # Voice / Audio
    (["voice","audio","sound","mic","record","speech","listen"],
     "📳"),

    # Fire / Trending
    (["fire","hot","viral","trending","epic","lit","savage"],
     "🔥"),
]

# Context-specific reactions for commands/events
CMD_REACTIONS = {
    "start":      "👍",
    "help":       "💡",
    "clear":      "✅",
    "stats":      "📈",
    "leaderboard":"🏆",
    "translate":  "📢",
    "summarize":  "📄",
    "ask":        "🤔",
    "mode":       "💡",
    "memory":     "👀",
    "admin":      "🔥",
    "broadcast":  "📢",
    "file":       "📄",
    "voice":      "🎵",
    "photo":      "👀",
    "error":      "💣",
    "success":    "✅",
    "ban":        "❌",
    "mute":       "❌",
    "pin":        "📢",
}

def pick_reaction(text):
    """Smart reaction based on message content."""
    low = text.lower()
    for keywords, emoji in REACTION_RULES:
        if any(k in low for k in keywords):
            return emoji
    # Random from friendly set
    friendly = ["👍", "🔥", "💯", "🙏",
                "🤔", "👀", "🎊", "✅"]
    return random.choice(friendly)

def react(chat_id, message_id, emoji):
    """
    Send reaction using raw Telegram Bot API.
    Most reliable method — bypasses pyTelegramBotAPI wrapper issues.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/setMessageReaction"
    try:
        r = http_requests.post(url, json={
            "chat_id":    chat_id,
            "message_id": message_id,
            "reaction":   [{"type": "emoji", "emoji": emoji}],
            "is_big":     False,
        }, timeout=5)
        data = r.json()
        if not data.get("ok"):
            desc = data.get("description", "")
            logger.debug(f"Reaction failed: {desc}")
            # Try fallback thumbs up if emoji invalid
            if "REACTION_INVALID" in desc or "invalid" in desc.lower():
                http_requests.post(url, json={
                    "chat_id":    chat_id,
                    "message_id": message_id,
                    "reaction":   [{"type": "emoji", "emoji": "\U0001f44d"}],
                    "is_big":     False,
                }, timeout=5)
    except Exception as e:
        logger.debug(f"Reaction error: {e}")

# Alias for backward compatibility
def send_reaction(chat_id, message_id, emoji):
    react(chat_id, message_id, emoji)

# ================================================================
#  FORMATTING
# ================================================================
def format_for_telegram(text):
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```(?:\w+)?\n?(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    return text.strip()

def safe_send(chat_id, text, reply_to=None, kb=None):
    if not text or not text.strip():
        text = "..."
    # Telegram max message length is 4096
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for i, chunk in enumerate(chunks):
        kwargs = {"parse_mode": "HTML"}
        if reply_to and i == 0:
            kwargs["reply_to_message_id"] = reply_to
        if kb and i == len(chunks) - 1:
            kwargs["reply_markup"] = kb
        try:
            bot.send_message(chat_id, chunk, **kwargs)
        except Exception:
            plain = re.sub(r"<[^>]+>", "", chunk)
            try:
                bot.send_message(chat_id, plain,
                    reply_to_message_id=(reply_to if i == 0 else None),
                    reply_markup=(kb if i == len(chunks)-1 else None))
            except Exception as e:
                logger.error(f"safe_send failed: {e}")

def send_typing(chat_id):
    try:
        bot.send_chat_action(chat_id, "typing")
    except Exception:
        pass

# ================================================================
#  AI CORE  —  with long-term memory + auto summary
# ================================================================

# In-memory last response store for "continue" button
last_responses = {}

def maybe_summarize(user_id):
    """If history is long, auto-summarize older messages to save context."""
    history = get_history(user_id)
    if len(history) < SUMMARY_AFTER:
        return
    # Summarize
    hist_text = "\n".join([f"{m['role'].upper()}: {m['content'][:300]}" for m in history[:-10]])
    try:
        resp = ai_client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": SUMMARY_PROMPT.format(history=hist_text)
            }],
            max_tokens=300,
        )
        summary = resp.choices[0].message.content.strip()
        save_summary(user_id, summary)
        # Keep only last 10 messages
        with db() as conn:
            ids = conn.execute(
                "SELECT id FROM messages WHERE user_id=? ORDER BY id DESC LIMIT 10",
                (user_id,)
            ).fetchall()
            if ids:
                keep_ids = [str(r[0]) for r in ids]
                conn.execute(
                    f"DELETE FROM messages WHERE user_id=? AND id NOT IN ({','.join(keep_ids)})",
                    (user_id,)
                )
            conn.commit()
        logger.info(f"Summarized history for user {user_id}")
    except Exception as e:
        logger.warning(f"Summary failed: {e}")

def get_ai_response(user_id, user_message, extra_system="", save_to_history=True):
    user = get_user(user_id)
    mode = user["mode"] if user else "normal"

    if save_to_history:
        save_message(user_id, "user", user_message)

    # Trigger summarization if needed
    threading.Thread(target=maybe_summarize, args=(user_id,), daemon=True).start()

    history = get_history(user_id)

    # Build system prompt
    system = BASE_SYSTEM
    if mode in MODE_PROMPTS and MODE_PROMPTS[mode]:
        system += f"\n\nCURRENT MODE: {MODE_PROMPTS[mode]}"
    if extra_system:
        system += f"\n\n{extra_system}"

    # Add long-term summary as context
    summary = get_summary(user_id)
    if summary:
        system += f"\n\nCONVERSATION SUMMARY (previous context):\n{summary}"

    messages = [{"role": "system", "content": system}] + history
    t0 = time.time()

    try:
        logger.info(f"AI call: user={user_id} mode={mode} history={len(history)}")
        resp = ai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        reply  = resp.choices[0].message.content.strip()
        tokens = getattr(resp.usage, "total_tokens", 0)
        rt     = time.time() - t0

        if save_to_history:
            save_message(user_id, "assistant", reply)

        update_user_stat(user_id, msg_count_delta=1, tokens_delta=tokens)
        track_analytics(user_id, tokens=tokens, response_time=rt)

        # Store last response for "continue" button
        last_responses[user_id] = {
            "last_reply": reply,
            "messages":   messages + [{"role": "assistant", "content": reply}],
        }

        logger.info(f"AI done: tokens={tokens} time={rt:.2f}s")
        return reply

    except Exception as e:
        err = str(e)
        log_error(user_id, err)
        logger.error(f"AI error user={user_id}: {type(e).__name__}: {err}")

        if "401" in err or "unauthorized" in err.lower():
            return "NARA_TOKEN invalid or expired. Please check your API key in Render environment variables."
        elif "429" in err or "rate" in err.lower():
            return "Too many requests. Please wait a moment and try again."
        elif "timeout" in err.lower():
            return "Request timed out. Please try again."
        elif "404" in err:
            return "AI model not found. Please contact admin."
        else:
            return f"AI Error: {err[:200]}\n\nPlease try again."

def continue_response(user_id):
    """Continue generating from where last response ended."""
    if user_id not in last_responses:
        return "No previous response to continue."
    data     = last_responses[user_id]
    messages = data["messages"] + [{"role": "user", "content": "Please continue from where you left off."}]
    try:
        resp   = ai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        reply  = resp.choices[0].message.content.strip()
        tokens = getattr(resp.usage, "total_tokens", 0)
        save_message(user_id, "assistant", f"[Continued] {reply}")
        update_user_stat(user_id, tokens_delta=tokens)
        track_analytics(user_id, tokens=tokens)
        last_responses[user_id]["messages"].append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"Continue failed: {e}"

# ================================================================
#  FILE PARSERS
# ================================================================
def parse_pdf(data):
    if not HAS_PDF:
        return "[PyPDF2 not installed]"
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        pages  = [p.extract_text() or "" for p in reader.pages[:20]]
        return "\n".join(pages).strip()[:12000] or "[No text extracted from PDF]"
    except Exception as e:
        return f"[PDF error: {e}]"

def parse_docx(data):
    if not HAS_DOCX:
        return "[python-docx not installed]"
    try:
        doc   = DocxDocument(io.BytesIO(data))
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(lines)[:12000]
    except Exception as e:
        return f"[DOCX error: {e}]"

def parse_excel(data):
    if not HAS_EXCEL:
        return "[openpyxl not installed]"
    try:
        wb     = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
        result = []
        for sheet in wb.sheetnames[:3]:
            ws   = wb[sheet]
            rows = list(ws.iter_rows(values_only=True))[:60]
            result.append(f"Sheet: {sheet}")
            for row in rows:
                result.append("\t".join(str(c) if c is not None else "" for c in row))
        return "\n".join(result)[:10000]
    except Exception as e:
        return f"[Excel error: {e}]"

def parse_csv(data):
    try:
        text = data.decode("utf-8", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))[:100]
        return f"CSV ({len(rows)} rows shown):\n" + "\n".join([",".join(r) for r in rows])
    except Exception as e:
        return f"[CSV error: {e}]"

def parse_json_file(data):
    try:
        obj = json.loads(data.decode("utf-8", errors="replace"))
        return json.dumps(obj, indent=2, ensure_ascii=False)[:10000]
    except Exception as e:
        return f"[JSON error: {e}]"

def parse_zip(data):
    try:
        result = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            result.append(f"ZIP Archive: {len(names)} files\n")
            for name in names[:30]:
                info = zf.getinfo(name)
                result.append(f"  - {name}  ({info.file_size:,} bytes)")
            for name in names[:5]:
                if any(name.endswith(e) for e in [".txt",".py",".js",".md",".json",".csv"]):
                    try:
                        content = zf.read(name).decode("utf-8", errors="replace")[:2000]
                        result.append(f"\n--- {name} ---\n{content}")
                    except Exception:
                        pass
        return "\n".join(result)[:10000]
    except Exception as e:
        return f"[ZIP error: {e}]"

CODE_EXTS = {
    ".py",".js",".ts",".java",".cpp",".c",".cs",".go",
    ".rb",".php",".swift",".kt",".rs",".html",".css",
    ".sh",".bash",".yaml",".yml",".toml",".xml",".sql",
    ".r",".m",".pl",".lua",".dart",".ex",".exs",
}

def parse_file(data, filename):
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":                    return parse_pdf(data),        "PDF Document"
    elif ext in (".docx",".doc"):        return parse_docx(data),       "Word Document"
    elif ext in (".xlsx",".xls"):        return parse_excel(data),      "Excel Spreadsheet"
    elif ext == ".csv":                  return parse_csv(data),        "CSV File"
    elif ext == ".json":                 return parse_json_file(data),  "JSON File"
    elif ext == ".zip":                  return parse_zip(data),        "ZIP Archive"
    elif ext in CODE_EXTS:
        return data.decode("utf-8", errors="replace")[:12000], f"Code File ({ext})"
    elif ext in (".txt",".md",".log",".text",".rst"):
        return data.decode("utf-8", errors="replace")[:12000], "Text File"
    else:
        try:
            return data.decode("utf-8", errors="replace")[:10000], f"File ({ext})"
        except Exception:
            return "", f"Unsupported ({ext})"

# ================================================================
#  VOICE — Speech to Text
# ================================================================
def transcribe_voice(file_bytes):
    if not HAS_VOICE:
        return None
    try:
        audio    = AudioSegment.from_ogg(io.BytesIO(file_bytes))
        wav_buf  = io.BytesIO()
        audio.export(wav_buf, format="wav")
        wav_buf.seek(0)
        rec      = sr.Recognizer()
        with sr.AudioFile(wav_buf) as source:
            audio_data = rec.record(source)
        return rec.recognize_google(audio_data)
    except Exception as e:
        logger.warning(f"Transcription error: {e}")
        return None

# ================================================================
#  LANGUAGE DETECTION
# ================================================================
COMMON_LANGS = {
    "hindi": "Hindi", "urdu": "Urdu", "english": "English",
    "spanish": "Spanish", "french": "French", "german": "German",
    "arabic": "Arabic", "chinese": "Chinese", "japanese": "Japanese",
    "korean": "Korean", "portuguese": "Portuguese", "russian": "Russian",
    "italian": "Italian", "turkish": "Turkish", "dutch": "Dutch",
    "bengali": "Bengali", "tamil": "Tamil", "telugu": "Telugu",
    "marathi": "Marathi", "gujarati": "Gujarati", "punjabi": "Punjabi",
}

def detect_language_prompt(text):
    """Ask AI to detect language."""
    return f"Detect the language of this text and respond with ONLY the language name, nothing else:\n\n{text[:200]}"

# ================================================================
#  KEYBOARDS
# ================================================================
def main_menu_kb():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("Normal Mode",    callback_data="mode_normal"),
        types.InlineKeyboardButton("Translate",      callback_data="mode_translate"),
        types.InlineKeyboardButton("Summarize",      callback_data="mode_summarize"),
        types.InlineKeyboardButton("Code Helper",    callback_data="mode_code"),
        types.InlineKeyboardButton("Essay Mode",     callback_data="mode_essay"),
        types.InlineKeyboardButton("Creative",       callback_data="mode_creative"),
        types.InlineKeyboardButton("My Stats",       callback_data="my_stats"),
        types.InlineKeyboardButton("Leaderboard",    callback_data="leaderboard"),
        types.InlineKeyboardButton("Memory Summary", callback_data="show_summary"),
        types.InlineKeyboardButton("Clear History",  callback_data="clear"),
    )
    return kb

def continue_kb():
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("Continue Response", callback_data="continue"))
    return kb

def mode_kb():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("Normal",    callback_data="mode_normal"),
        types.InlineKeyboardButton("Translate", callback_data="mode_translate"),
        types.InlineKeyboardButton("Summarize", callback_data="mode_summarize"),
        types.InlineKeyboardButton("Code",      callback_data="mode_code"),
        types.InlineKeyboardButton("Essay",     callback_data="mode_essay"),
        types.InlineKeyboardButton("Creative",  callback_data="mode_creative"),
    )
    return kb

def translate_kb():
    kb = types.InlineKeyboardMarkup(row_width=3)
    langs = ["Hindi","Urdu","English","Spanish","French","German",
             "Arabic","Chinese","Japanese","Korean","Russian","Turkish"]
    for lang in langs:
        kb.add(types.InlineKeyboardButton(lang, callback_data=f"translate_to_{lang}"))
    return kb

# ================================================================
#  FLASK
# ================================================================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    today = datetime.now().strftime("%Y-%m-%d")
    return (
        f"Bot running | "
        f"Users: {len(_analytics_cache['total_users'])} | "
        f"Messages: {_analytics_cache['total_messages']}"
    ), 200

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    update = telebot.types.Update.de_json(request.get_data().decode("utf-8"))
    bot.process_new_updates([update])
    return "OK", 200

# ================================================================
#  COMMAND HANDLERS
# ================================================================
def is_admin(user_id):
    return user_id in ADMIN_IDS

@bot.message_handler(commands=["start"])
def cmd_start(msg):
    uid  = msg.from_user.id
    name = msg.from_user.first_name or "there"
    ensure_user(uid, name)
    react(msg.chat.id, msg.message_id, CMD_REACTIONS["start"])
    text = (
        f"Hello <b>{name}!</b>\n\n"
        "I am your <b>Advanced AI Assistant</b>.\n\n"
        "<b>What I can do:</b>\n"
        "- Answer <b>unlimited questions</b> with full memory\n"
        "- Remember <b>all your conversations</b> across sessions\n"
        "- Read <b>PDF, DOCX, Excel, CSV, JSON, ZIP</b> files\n"
        "- <b>Translate</b> text in 100+ languages\n"
        "- <b>Voice messages</b> to text\n"
        "- <b>Summarize</b> long content\n"
        "- <b>Code</b> help and debugging\n"
        "- <b>Creative</b> writing\n\n"
        "Select a mode below or just start typing!"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML", reply_markup=main_menu_kb())


@bot.message_handler(commands=["help"])
def cmd_help(msg):
    react(msg.chat.id, msg.message_id, CMD_REACTIONS["help"])
    text = (
        "<b>Commands:</b>\n\n"
        "/start - Welcome screen\n"
        "/mode - Switch AI mode\n"
        "/clear - Clear chat history\n"
        "/memory - Show your memory summary\n"
        "/stats - Your usage stats\n"
        "/ask [question] - Quick question\n"
        "/translate [text] - Translate text\n"
        "/detect [text] - Detect language\n"
        "/summarize [text] - Summarize text\n"
        "/leaderboard - Top users\n"
        "/setlang [language] - Set your language\n"
        "/help - This message\n"
    )
    if is_admin(msg.from_user.id):
        text += (
            "\n<b>Admin Commands:</b>\n"
            "/admin - Admin dashboard\n"
            "/errors - Error logs\n"
            "/broadcast [msg] - Message all users\n"
        )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")


@bot.message_handler(commands=["mode"])
def cmd_mode(msg):
    bot.send_message(msg.chat.id, "Select AI Mode:", reply_markup=mode_kb())


@bot.message_handler(commands=["clear"])
def cmd_clear(msg):
    uid = msg.from_user.id
    clear_history(uid)
    last_responses.pop(uid, None)
    react(msg.chat.id, msg.message_id, CMD_REACTIONS["clear"])
    bot.send_message(msg.chat.id,
        "History cleared! Memory reset. Fresh start.",
        parse_mode="HTML")


@bot.message_handler(commands=["memory"])
def cmd_memory(msg):
    uid     = msg.from_user.id
    summary = get_summary(uid)
    history = get_history(uid)
    if summary:
        text = f"<b>Your Memory Summary:</b>\n\n{summary}\n\n<i>({len(history)} messages in active history)</i>"
    elif history:
        text = f"<b>No summary yet</b> — {len(history)} messages in history.\n\nSummary is auto-generated after {SUMMARY_AFTER} messages."
    else:
        text = "No conversation history yet. Start chatting!"
    bot.send_message(msg.chat.id, text, parse_mode="HTML")


@bot.message_handler(commands=["stats"])
def cmd_stats(msg):
    uid  = msg.from_user.id
    ensure_user(uid)
    user = get_user(uid)
    hist = get_history(uid)
    text = (
        "<b>Your Stats:</b>\n\n"
        f"- Messages sent: <b>{user['msg_count']}</b>\n"
        f"- Tokens used: <b>{user['tokens_used']:,}</b>\n"
        f"- Using since: <b>{user['first_seen']}</b>\n"
        f"- Current mode: <b>{user['mode'].capitalize()}</b>\n"
        f"- Language: <b>{user['language']}</b>\n"
        f"- History: <b>{len(hist)} messages</b>"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")


@bot.message_handler(commands=["leaderboard"])
def cmd_leaderboard(msg):
    top    = get_leaderboard(10)
    medals = ["1st","2nd","3rd"] + [f"{i}th" for i in range(4,11)]
    rows   = ["<b>Top Users:</b>\n"]
    for i, (uid, name, count, tokens) in enumerate(top):
        display = name or f"User{uid}"
        rows.append(f"{medals[i]} {display} - <b>{count}</b> msgs ({tokens:,} tokens)")
    bot.send_message(msg.chat.id, "\n".join(rows), parse_mode="HTML")


@bot.message_handler(commands=["ask"])
def cmd_ask(msg):
    q = msg.text.partition(" ")[2].strip()
    if not q:
        bot.send_message(msg.chat.id, "Usage: /ask Your question here")
        return
    uid = msg.from_user.id
    ensure_user(uid, msg.from_user.first_name or "")
    send_typing(msg.chat.id)
    react(msg.chat.id, msg.message_id, CMD_REACTIONS["ask"])
    reply = get_ai_response(uid, q)
    safe_send(msg.chat.id, format_for_telegram(reply),
              reply_to=msg.message_id, kb=continue_kb())


@bot.message_handler(commands=["translate"])
def cmd_translate(msg):
    txt = msg.text.partition(" ")[2].strip()
    if not txt:
        bot.send_message(msg.chat.id,
            "Usage: /translate [text]\n\nOr choose a target language:",
            reply_markup=translate_kb())
        return
    uid = msg.from_user.id
    ensure_user(uid)
    send_typing(msg.chat.id)
    reply = get_ai_response(uid,
        f"Detect the language and translate to English:\n\n{txt}",
        save_to_history=False)
    safe_send(msg.chat.id, format_for_telegram(reply), reply_to=msg.message_id)


@bot.message_handler(commands=["detect"])
def cmd_detect(msg):
    txt = msg.text.partition(" ")[2].strip()
    if not txt:
        bot.send_message(msg.chat.id, "Usage: /detect [text to detect language of]")
        return
    uid = msg.from_user.id
    ensure_user(uid)
    send_typing(msg.chat.id)
    reply = get_ai_response(uid,
        f"Detect the language of this text. Tell the language name, confidence, and any interesting details:\n\n{txt}",
        save_to_history=False)
    safe_send(msg.chat.id, format_for_telegram(reply), reply_to=msg.message_id)


@bot.message_handler(commands=["summarize"])
def cmd_summarize(msg):
    txt = msg.text.partition(" ")[2].strip()
    if not txt:
        bot.send_message(msg.chat.id, "Usage: /summarize [text here]")
        return
    uid = msg.from_user.id
    ensure_user(uid)
    send_typing(msg.chat.id)
    reply = get_ai_response(uid,
        f"Summarize in 3-5 bullet points:\n\n{txt}",
        save_to_history=False)
    safe_send(msg.chat.id, format_for_telegram(reply), reply_to=msg.message_id)


@bot.message_handler(commands=["setlang"])
def cmd_setlang(msg):
    lang = msg.text.partition(" ")[2].strip()
    if not lang:
        bot.send_message(msg.chat.id,
            "Usage: /setlang Hindi\n\nSupported: Hindi, Urdu, English, Spanish, French, German, Arabic, Chinese, Japanese, Korean, Russian, Turkish, Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi, and 100+ more")
        return
    uid = msg.from_user.id
    ensure_user(uid)
    set_user_language(uid, lang.capitalize())
    bot.send_message(msg.chat.id,
        f"Language set to <b>{lang.capitalize()}</b>! I will now respond in {lang.capitalize()}.",
        parse_mode="HTML")


# ── ADMIN COMMANDS ───────────────────────────────────────────────
@bot.message_handler(commands=["admin"])
def cmd_admin(msg):
    if not is_admin(msg.from_user.id):
        bot.send_message(msg.chat.id, "Admin only.")
        return
    today = datetime.now().strftime("%Y-%m-%d")
    rt    = _analytics_cache["response_times"]
    avg   = round(sum(rt)/len(rt), 2) if rt else 0
    mn    = round(min(rt), 2) if rt else 0
    mx    = round(max(rt), 2) if rt else 0
    text  = (
        "<b>Admin Dashboard:</b>\n\n"
        f"- Total users: <b>{len(_analytics_cache['total_users'])}</b>\n"
        f"- Active today: <b>{len(_analytics_cache['active_today'])}</b>\n"
        f"- Total messages: <b>{_analytics_cache['total_messages']:,}</b>\n"
        f"- Today messages: <b>{_analytics_cache['daily_messages'].get(today,0)}</b>\n"
        f"- Total tokens: <b>{_analytics_cache['total_tokens']:,}</b>\n"
        f"- Model: <b>{MODEL}</b>\n"
        f"- Avg response: <b>{avg}s</b>\n"
        f"- Fastest: <b>{mn}s</b>\n"
        f"- Slowest: <b>{mx}s</b>\n"
        f"- Errors: <b>{_analytics_cache['errors_count']}</b>"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")


@bot.message_handler(commands=["errors"])
def cmd_errors(msg):
    if not is_admin(msg.from_user.id):
        bot.send_message(msg.chat.id, "Admin only.")
        return
    errors = get_errors(10)
    if not errors:
        bot.send_message(msg.chat.id, "No errors logged!")
        return
    rows = ["<b>Last 10 Errors:</b>\n"]
    for uid, error, created_at in errors:
        rows.append(f"[{created_at[:16]}] User {uid}:\n<code>{error[:100]}</code>")
    bot.send_message(msg.chat.id, "\n\n".join(rows), parse_mode="HTML")


@bot.message_handler(commands=["broadcast"])
def cmd_broadcast(msg):
    if not is_admin(msg.from_user.id):
        bot.send_message(msg.chat.id, "Admin only.")
        return
    text = msg.text.partition(" ")[2].strip()
    if not text:
        bot.send_message(msg.chat.id, "Usage: /broadcast Your message here")
        return
    users = get_all_users()
    sent  = 0
    for uid in users:
        try:
            bot.send_message(uid,
                f"<b>Announcement:</b>\n\n{text}",
                parse_mode="HTML")
            sent += 1
            time.sleep(0.05)
        except Exception:
            pass
    bot.send_message(msg.chat.id, f"Sent to <b>{sent}</b> users.", parse_mode="HTML")


# ── Group Admin Tools ────────────────────────────────────────────
@bot.message_handler(commands=["ban"])
def cmd_ban(msg):
    if msg.chat.type == "private" or not msg.reply_to_message:
        return
    try:
        bot.ban_chat_member(msg.chat.id, msg.reply_to_message.from_user.id)
        bot.send_message(msg.chat.id, "User banned.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"Cannot ban: {e}")

@bot.message_handler(commands=["unban"])
def cmd_unban(msg):
    if msg.chat.type == "private" or not msg.reply_to_message:
        return
    try:
        bot.unban_chat_member(msg.chat.id, msg.reply_to_message.from_user.id)
        bot.send_message(msg.chat.id, "User unbanned.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"Cannot unban: {e}")

@bot.message_handler(commands=["mute"])
def cmd_mute(msg):
    if msg.chat.type == "private" or not msg.reply_to_message:
        return
    try:
        bot.restrict_chat_member(
            msg.chat.id, msg.reply_to_message.from_user.id,
            types.ChatPermissions(can_send_messages=False),
            until_date=int(time.time()) + 3600
        )
        bot.send_message(msg.chat.id, "User muted for 1 hour.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"Cannot mute: {e}")

@bot.message_handler(commands=["unmute"])
def cmd_unmute(msg):
    if msg.chat.type == "private" or not msg.reply_to_message:
        return
    try:
        bot.restrict_chat_member(
            msg.chat.id, msg.reply_to_message.from_user.id,
            types.ChatPermissions(
                can_send_messages=True, can_send_media_messages=True,
                can_send_polls=True, can_send_other_messages=True,
            )
        )
        bot.send_message(msg.chat.id, "User unmuted.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"Cannot unmute: {e}")

@bot.message_handler(commands=["kick"])
def cmd_kick(msg):
    if msg.chat.type == "private" or not msg.reply_to_message:
        return
    try:
        uid = msg.reply_to_message.from_user.id
        bot.ban_chat_member(msg.chat.id, uid)
        bot.unban_chat_member(msg.chat.id, uid)
        bot.send_message(msg.chat.id, "User kicked.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"Cannot kick: {e}")

@bot.message_handler(commands=["pin"])
def cmd_pin(msg):
    if msg.chat.type == "private" or not msg.reply_to_message:
        return
    try:
        bot.pin_chat_message(msg.chat.id, msg.reply_to_message.message_id)
        bot.send_message(msg.chat.id, "Message pinned.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"Cannot pin: {e}")

# ================================================================
#  CALLBACK HANDLER
# ================================================================
@bot.callback_query_handler(func=lambda c: True)
def handle_callback(call):
    uid  = call.from_user.id
    data = call.data
    ensure_user(uid, call.from_user.first_name or "")

    # Mode switch
    if data.startswith("mode_"):
        mode = data[5:]
        set_user_mode(uid, mode)
        labels = {
            "normal":    "Normal Mode",
            "translate": "Translate Mode",
            "summarize": "Summarize Mode",
            "code":      "Code Mode",
            "essay":     "Essay Mode",
            "creative":  "Creative Mode",
        }
        label = labels.get(mode, mode.capitalize())
        bot.answer_callback_query(call.id, f"{label} activated!")
        safe_send(call.message.chat.id,
            f"Switched to <b>{label}</b>! Send your message.")

    # Continue response
    elif data == "continue":
        bot.answer_callback_query(call.id, "Continuing...")
        send_typing(call.message.chat.id)
        reply = continue_response(uid)
        safe_send(call.message.chat.id,
            format_for_telegram(reply), kb=continue_kb())

    # Clear history
    elif data == "clear":
        clear_history(uid)
        last_responses.pop(uid, None)
        bot.answer_callback_query(call.id, "History cleared!")
        safe_send(call.message.chat.id, "History and memory cleared! Fresh start.")

    # My stats
    elif data == "my_stats":
        user = get_user(uid)
        hist = get_history(uid)
        bot.answer_callback_query(call.id)
        if user:
            safe_send(call.message.chat.id,
                f"<b>Your Stats:</b>\n\n"
                f"- Messages: <b>{user['msg_count']}</b>\n"
                f"- Tokens: <b>{user['tokens_used']:,}</b>\n"
                f"- Since: <b>{user['first_seen']}</b>\n"
                f"- Mode: <b>{user['mode'].capitalize()}</b>\n"
                f"- Language: <b>{user['language']}</b>\n"
                f"- History: <b>{len(hist)} msgs</b>")

    # Leaderboard
    elif data == "leaderboard":
        top    = get_leaderboard(10)
        medals = ["1st","2nd","3rd"] + [f"{i}th" for i in range(4,11)]
        rows   = ["<b>Top Users:</b>\n"]
        for i, (u, name, count, tokens) in enumerate(top):
            display = name or f"User{u}"
            rows.append(f"{medals[i]} {display} - <b>{count}</b> msgs")
        bot.answer_callback_query(call.id)
        safe_send(call.message.chat.id, "\n".join(rows))

    # Show memory summary
    elif data == "show_summary":
        summary = get_summary(uid)
        history = get_history(uid)
        bot.answer_callback_query(call.id)
        if summary:
            safe_send(call.message.chat.id,
                f"<b>Memory Summary:</b>\n\n{summary}\n\n<i>({len(history)} active messages)</i>")
        else:
            safe_send(call.message.chat.id,
                f"No summary yet. {len(history)} messages in history.\nAuto-summary after {SUMMARY_AFTER} messages.")

    # Translate to specific language
    elif data.startswith("translate_to_"):
        lang = data.replace("translate_to_", "")
        bot.answer_callback_query(call.id, f"Translate to {lang} mode!")
        safe_send(call.message.chat.id,
            f"Now send the text you want to translate to <b>{lang}</b>.",
            kb=None)
        # Store pending translation target
        last_responses[uid] = last_responses.get(uid, {})
        last_responses[uid]["pending_translate"] = lang

    else:
        bot.answer_callback_query(call.id)

# ================================================================
#  FILE / DOCUMENT HANDLER
# ================================================================
@bot.message_handler(content_types=["document"])
def handle_document(msg):
    uid      = msg.from_user.id
    chat_id  = msg.chat.id
    filename = msg.document.file_name or "file"
    caption  = msg.caption or ""

    ensure_user(uid, msg.from_user.first_name or "")
    react(chat_id, msg.message_id, CMD_REACTIONS["file"])
    send_typing(chat_id)
    wait = bot.send_message(chat_id,
        f"Reading <b>{filename}</b>...", parse_mode="HTML")

    try:
        file_info  = bot.get_file(msg.document.file_id)
        file_bytes = bot.download_file(file_info.file_path)
    except Exception as e:
        bot.edit_message_text(
            f"Could not download: {e}", chat_id, wait.message_id)
        return

    parsed, label = parse_file(file_bytes, filename)
    if not parsed:
        bot.edit_message_text(
            f"Could not read <b>{filename}</b>.",
            chat_id, wait.message_id, parse_mode="HTML")
        return

    question = caption if caption else f"Analyze this {label} thoroughly and give a detailed summary with key insights, important data, and conclusions."
    prompt   = f"[{label}: {filename}]\n\n{parsed[:9000]}\n\nUser request: {question}"

    bot.edit_message_text(
        f"Analyzing <b>{filename}</b>...",
        chat_id, wait.message_id, parse_mode="HTML")

    reply = get_ai_response(uid, prompt,
        extra_system=f"User uploaded a {label} named '{filename}'. Analyze it thoroughly. Use <b>bold</b> for key points. Structure your response clearly.")
    bot.delete_message(chat_id, wait.message_id)
    safe_send(chat_id, format_for_telegram(reply),
              reply_to=msg.message_id, kb=continue_kb())


# ================================================================
#  VOICE HANDLER
# ================================================================
@bot.message_handler(content_types=["voice"])
def handle_voice(msg):
    uid     = msg.from_user.id
    chat_id = msg.chat.id
    ensure_user(uid, msg.from_user.first_name or "")
    react(chat_id, msg.message_id, CMD_REACTIONS["voice"])

    if not HAS_VOICE:
        safe_send(chat_id,
            "Voice received! Please <b>type your message</b> for now.",
            reply_to=msg.message_id)
        return

    send_typing(chat_id)
    wait = bot.send_message(chat_id, "Transcribing voice message...")
    try:
        file_info  = bot.get_file(msg.voice.file_id)
        file_bytes = bot.download_file(file_info.file_path)
        text       = transcribe_voice(file_bytes)
        if text:
            bot.edit_message_text(
                f"<b>You said:</b> {text}", chat_id, wait.message_id,
                parse_mode="HTML")
            reply = get_ai_response(uid, text)
            safe_send(chat_id, format_for_telegram(reply),
                      reply_to=msg.message_id, kb=continue_kb())
        else:
            bot.edit_message_text(
                "Could not transcribe. Please try again or type your message.",
                chat_id, wait.message_id)
    except Exception as e:
        bot.edit_message_text(f"Voice error: {e}", chat_id, wait.message_id)


# ================================================================
#  PHOTO HANDLER
# ================================================================
@bot.message_handler(content_types=["photo"])
def handle_photo(msg):
    ensure_user(msg.from_user.id, msg.from_user.first_name or "")
    react(msg.chat.id, msg.message_id, CMD_REACTIONS["photo"])
    safe_send(msg.chat.id,
        "Image received! I cannot analyze images yet.\n\nIf your image contains text, please <b>type it</b> and I will help.",
        reply_to=msg.message_id)


# ================================================================
#  MAIN TEXT HANDLER
# ================================================================
@bot.message_handler(func=lambda m: m.text and not m.text.startswith("/"))
def handle_text(msg):
    uid       = msg.from_user.id
    chat_id   = msg.chat.id
    user_text = msg.text.strip()

    ensure_user(uid, msg.from_user.first_name or "")
    logger.info(f"User {uid}: {user_text[:60]}")

    send_typing(chat_id)
    react(chat_id, msg.message_id, pick_reaction(user_text))

    # Check if user has pending translate target
    extra = ""
    if uid in last_responses and last_responses[uid].get("pending_translate"):
        lang = last_responses[uid].pop("pending_translate")
        extra = f"Translate the following text to {lang}. Also mention the source language detected."

    # Check user language preference
    user = get_user(uid)
    if user and user["language"] != "auto":
        extra += f"\n\nAlways respond in {user['language']}."

    reply     = get_ai_response(uid, user_text, extra_system=extra)
    formatted = format_for_telegram(reply)
    safe_send(chat_id, formatted, reply_to=msg.message_id, kb=continue_kb())


@bot.message_handler(content_types=["sticker","video","audio","video_note"])
def handle_other(msg):
    safe_send(msg.chat.id,
        "Send <b>text messages</b>, <b>voice messages</b>, or <b>files</b> (PDF, DOCX, Excel, CSV, JSON, ZIP, code files).",
        reply_to=msg.message_id)


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    PORT       = int(os.environ.get("PORT", 5000))
    RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/{BOT_TOKEN}"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook set: {webhook_url}")
        app.run(host="0.0.0.0", port=PORT)
    else:
        logger.info("Polling mode (local)...")
        bot.remove_webhook()
        threading.Thread(target=bot.infinity_polling, daemon=True).start()
        app.run(host="0.0.0.0", port=PORT)
