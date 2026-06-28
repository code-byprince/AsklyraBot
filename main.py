# ═══════════════════════════════════════════════════════════════════════════════
#  ADVANCED TELEGRAM AI BOT  — DeepSeek via HuggingFace Router
#  Features: File analysis, Voice, Group admin, Analytics, Unlimited history
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
HF_TOKEN   = os.environ.get("HF_TOKEN")
ADMIN_IDS  = [int(x) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip().isdigit()]

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not set!")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set!")

# ── Clients ────────────────────────────────────────────────────────────────────
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)

ai_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

MODEL      = "deepseek-ai/DeepSeek-V3-0324:novita"
MAX_TOKENS = 2048

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

MAX_HISTORY = 40   # keep 40 messages — smart trimming handles token limits

# ── Smart history trim — keeps first 2 + last N messages ──────────────────────
def trim_history(history: list) -> list:
    if len(history) <= MAX_HISTORY:
        return history
    return history[:2] + history[-(MAX_HISTORY - 2):]

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
    "translate": "Translate every message the user sends to English (unless they specify another target language). Always identify the source language.",
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
    analytics["total_tokens"] += count
    if user_id and user_id in user_stats:
        user_stats[user_id]["tokens_used"] += count

def track_error(user_id: int, error: str):
    analytics["errors"].append({
        "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "error":   str(error)[:200],
    })
    analytics["errors"] = analytics["errors"][-100:]   # keep last 100

def track_response_time(seconds: float):
    analytics["response_times"].append(round(seconds, 3))
    analytics["response_times"] = analytics["response_times"][-500:]

def leaderboard() -> list:
    return sorted(
        [(uid, d["count"], d.get("name","?"), d.get("tokens_used",0))
         for uid, d in user_stats.items()],
        key=lambda x: x[1], reverse=True
    )[:10]

# ═══════════════════════════════════════════════════════════════════════════════
#  AI CORE  — unlimited history with smart trimming
# ═══════════════════════════════════════════════════════════════════════════════
def get_ai_response(user_id: int, user_message: str, extra_system: str = "") -> str:
    if user_id not in user_histories:
        user_histories[user_id] = []

    mode    = user_modes.get(user_id, "normal")
    history = user_histories[user_id]
    history.append({"role": "user", "content": user_message})
    history = trim_history(history)
    user_histories[user_id] = history

    system = BASE_SYSTEM
    if MODE_PROMPTS.get(mode):
        system += "\n\nMODE: " + MODE_PROMPTS[mode]
    if extra_system:
        system += "\n\n" + extra_system

    messages = [{"role": "system", "content": system}] + history

    t0 = time.time()
    try:
        resp = ai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        reply  = resp.choices[0].message.content.strip()
        tokens = getattr(resp.usage, "total_tokens", 0)
        history.append({"role": "assistant", "content": reply})
        track_tokens(tokens, user_id)
        analytics["model_usage"][MODEL] += 1
        track_response_time(time.time() - t0)
        return reply
    except Exception as e:
        track_error(user_id, e)
        logger.error(f"AI error for {user_id}: {e}")
        return "⚠️ AI service is temporarily unavailable. Please try again in a moment."

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
        f"👋 <b>Hello, {name}!</b>\n\n"
        "I'm your <b>Advanced AI Assistant</b> powered by <b>DeepSeek</b>.\n\n"
        "<b>What I can do:</b>\n"
        "• Answer <b>unlimited questions</b> with full memory\n"
        "• <b>Read & analyze</b> PDF, DOCX, Excel, CSV, JSON, ZIP, code files\n"
        "• <b>Translate</b> text in any language\n"
        "• <b>Summarize</b> long content\n"
        "• <b>Code review</b> and debugging\n"
        "• <b>Voice messages</b> (speech to text)\n\n"
        "Choose a mode or just start typing! 👇"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML", reply_markup=main_menu_kb())


@bot.message_handler(commands=["help"])
def cmd_help(msg):
    send_reaction(msg.chat.id, msg.message_id, "💡")
    text = (
        "<b>📌 Commands:</b>\n\n"
        "/start — Welcome + mode selector\n"
        "/mode — Switch AI mode\n"
        "/clear — Clear your history\n"
        "/stats — Your personal stats\n"
        "/ask [question] — Quick AI question\n"
        "/translate [text] — Translate text\n"
        "/summarize [text] — Summarize text\n"
        "/leaderboard — Top users\n"
        "/help — This message\n"
    )
    if is_admin(msg.from_user.id):
        text += (
            "\n<b>🔐 Admin    (["wow", "amazing", "incredible", "unbelievable"],     "🤩"),
    (["food", "khana", "eat", "recipe", "hungry"],         "🍕"),
    (["music", "song", "gaana", "listen", "playlist"],     "🎵"),
    (["sports", "cricket", "football", "game", "match"],   "⚽"),
    (["weather", "rain", "sunny", "temperature", "mausam"],"⛅"),
]

DEFAULT_REACTIONS = ["👍", "🔥", "💯", "⚡", "🤩", "👏"]

def pick_reaction(text: str) -> str:
    lowered = text.lower()
    for keywords, emoji in REACTION_MAP:
        if any(kw in lowered for kw in keywords):
            return emoji
    return random.choice(DEFAULT_REACTIONS)

def send_reaction(chat_id: int, message_id: int, emoji: str):
    """Send reaction using raw Telegram API — more reliable than pyTelegramBotAPI wrapper."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/setMessageReaction"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "reaction": [{"type": "emoji", "emoji": emoji}],
        "is_big": False,
    }
    try:
        resp = http_requests.post(url, json=payload, timeout=5)
        data = resp.json()
        if not data.get("ok"):
            logger.warning(f"Reaction failed: {data.get('description')}")
    except Exception as e:
        logger.warning(f"Reaction error: {e}")

# ── Text Formatting ───────────────────────────────────────────────────────────
def clean_ai_response(text: str) -> str:
    """
    Remove markdown headers (#, ##, ###) and convert **bold** to *bold*
    so Telegram HTML mode can render it. Returns HTML-safe string.
    """
    # Remove heading markers (# ## ###) — replace with plain text + newline
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Convert **bold** → <b>bold</b>
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)

    # Convert *italic* (single) → <i>italic</i>  — only if not already a bold tag
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)

    # Convert `inline code` → <code>inline code</code>
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)

    # Convert triple backtick code blocks → <pre>
    text = re.sub(r"```(?:\w+)?\n?(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)

    # Escape remaining & < > that are NOT part of our HTML tags
    # (We inserted our own tags so we protect them)
    # This is tricky — we use a safe approach: escape first, then re-insert tags
    return text

def format_for_telegram(raw: str) -> str:
    """Full pipeline: clean AI markdown → Telegram HTML."""
    return clean_ai_response(raw)

# ── Utility ───────────────────────────────────────────────────────────────────
def send_typing(chat_id: int):
    try:
        bot.send_chat_action(chat_id, "typing")
    except Exception:
        pass

def update_stats(user_id: int):
    if user_id not in user_stats:
        user_stats[user_id] = {"count": 0, "first_seen": datetime.now().strftime("%Y-%m-%d")}
    user_stats[user_id]["count"] += 1

# ── AI Core ───────────────────────────────────────────────────────────────────
def get_ai_response(user_id: int, user_message: str) -> str:
    if user_id not in user_histories:
        user_histories[user_id] = []

    mode    = user_modes.get(user_id, "normal")
    history = user_histories[user_id]
    history.append({"role": "user", "content": user_message})

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        user_histories[user_id] = history

    system = BASE_SYSTEM
    if mode in MODE_PROMPTS and MODE_PROMPTS[mode]:
        system += "\n\nCURRENT MODE: " + MODE_PROMPTS[mode]

    messages = [{"role": "system", "content": system}] + history

    try:
        response = ai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"AI API error: {e}")
        return "Sorry, something went wrong. Please try again in a moment."

# ── Inline Keyboard Helpers ───────────────────────────────────────────────────
def main_menu_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("🧠 Normal Mode",     callback_data="mode_normal"),
        types.InlineKeyboardButton("🌐 Translate Mode",  callback_data="mode_translate"),
        types.InlineKeyboardButton("📝 Summarize Mode",  callback_data="mode_summarize"),
        types.InlineKeyboardButton("💻 Code Mode",       callback_data="mode_code"),
        types.InlineKeyboardButton("📊 My Stats",        callback_data="stats"),
        types.InlineKeyboardButton("🗑️ Clear History",   callback_data="clear"),
    )
    return kb

# ── Command Handlers ──────────────────────────────────────────────────────────
@bot.message_handler(commands=["start"])
def handle_start(message):
    user_id = message.from_user.id
    name    = message.from_user.first_name or "there"
    update_stats(user_id)
    send_reaction(message.chat.id, message.message_id, "👋")

    text = (
        f"👋 <b>Hello, {name}!</b>\n\n"
        "I'm your <b>AI Assistant</b> powered by DeepSeek.\n\n"
        "I can help you with:\n"
        "• Answering <b>any question</b>\n"
        "• <b>Translating</b> text to any language\n"
        "• <b>Summarizing</b> long content\n"
        "• <b>Coding</b> help and debugging\n"
        "• <b>Writing</b> and editing\n\n"
        "Choose a mode below or just start typing!"
    )
    bot.send_message(message.chat.id, text, parse_mode="HTML", reply_markup=main_menu_keyboard())


@bot.message_handler(commands=["help"])
def handle_help(message):
    send_reaction(message.chat.id, message.message_id, "💡")
    text = (
        "<b>Available Commands:</b>\n\n"
        "/start — Welcome screen + mode selector\n"
        "/mode — Switch AI mode\n"
        "/clear — Clear conversation history\n"
        "/stats — Your usage stats\n"
        "/ask [question] — Quick question\n"
        "/translate [text] — Translate text\n"
        "/summarize [text] — Summarize text\n"
        "/help — Show this message\n\n"
        "<b>Modes:</b>\n"
        "• Normal — General AI assistant\n"
        "• Translate — Auto-translate messages\n"
        "• Summarize — Summarize any text\n"
        "• Code — Coding assistant"
    )
    bot.send_message(message.chat.id, text, parse_mode="HTML")


@bot.message_handler(commands=["mode"])
def handle_mode(message):
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("🧠 Normal",    callback_data="mode_normal"),
        types.InlineKeyboardButton("🌐 Translate", callback_data="mode_translate"),
        types.InlineKeyboardButton("📝 Summarize", callback_data="mode_summarize"),
        types.InlineKeyboardButton("💻 Code",      callback_data="mode_code"),
    )
    bot.send_message(message.chat.id, "Select a mode:", reply_markup=kb)


@bot.message_handler(commands=["clear"])
def handle_clear(message):
    user_histories[message.from_user.id] = []
    send_reaction(message.chat.id, message.message_id, "👍")
    bot.send_message(message.chat.id, "🗑️ <b>Conversation cleared!</b> Starting fresh.", parse_mode="HTML")


@bot.message_handler(commands=["stats"])
def handle_stats(message):
    user_id = message.from_user.id
    update_stats(user_id)
    s     = user_stats.get(user_id, {})
    count = s.get("count", 0)
    since = s.get("first_seen", "today")
    mode  = user_modes.get(user_id, "normal").capitalize()
    hist  = len(user_histories.get(user_id, []))

    text = (
        "📊 <b>Your Stats:</b>\n\n"
        f"• Messages sent: <b>{count}</b>\n"
        f"• Using bot since: <b>{since}</b>\n"
        f"• Current mode: <b>{mode}</b>\n"
        f"• History length: <b>{hist} messages</b>"
    )
    bot.send_message(message.chat.id, text, parse_mode="HTML")


@bot.message_handler(commands=["ask"])
def handle_ask(message):
    question = message.text.partition(" ")[2].strip()
    if not question:
        bot.send_message(message.chat.id, "Usage: /ask What is quantum computing?")
        return
    send_typing(message.chat.id)
    send_reaction(message.chat.id, message.message_id, "🤔")
    reply = get_ai_response(message.from_user.id, question)
    formatted = format_for_telegram(reply)
    bot.send_message(message.chat.id, formatted, parse_mode="HTML")


@bot.message_handler(commands=["translate"])
def handle_translate(message):
    text_to_translate = message.text.partition(" ")[2].strip()
    if not text_to_translate:
        bot.send_message(message.chat.id, "Usage: /translate Aap kaise hain?")
        return
    send_typing(message.chat.id)
    send_reaction(message.chat.id, message.message_id, "🌐")
    prompt = f"Translate this to English and tell me what language it was: {text_to_translate}"
    reply  = get_ai_response(message.from_user.id, prompt)
    bot.send_message(message.chat.id, format_for_telegram(reply), parse_mode="HTML")


@bot.message_handler(commands=["summarize"])
def handle_summarize(message):
    content = message.text.partition(" ")[2].strip()
    if not content:
        bot.send_message(message.chat.id, "Usage: /summarize [paste your long text here]")
        return
    send_typing(message.chat.id)
    send_reaction(message.chat.id, message.message_id, "📝")
    prompt = f"Summarize this in 3-5 bullet points:\n\n{content}"
    reply  = get_ai_response(message.from_user.id, prompt)
    bot.send_message(message.chat.id, format_for_telegram(reply), parse_mode="HTML")


# ── Callback Query Handler ────────────────────────────────────────────────────
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    user_id = call.from_user.id
    data    = call.data

    if data.startswith("mode_"):
        mode = data.replace("mode_", "")
        user_modes[user_id] = mode
        mode_names = {
            "normal":    "🧠 Normal Mode",
            "translate": "🌐 Translate Mode",
            "summarize": "📝 Summarize Mode",
            "code":      "💻 Code Mode",
        }
        label = mode_names.get(mode, mode.capitalize())
        bot.answer_callback_query(call.id, f"Switched to {label}")
        bot.send_message(
            call.message.chat.id,
            f"Switched to <b>{label}</b>!\n\nNow just send your message.",
            parse_mode="HTML",
        )

    elif data == "clear":
        user_histories[user_id] = []
        bot.answer_callback_query(call.id, "History cleared!")
        bot.send_message(call.message.chat.id, "🗑️ <b>History cleared!</b>", parse_mode="HTML")

    elif data == "stats":
        update_stats(user_id)
        s     = user_stats.get(user_id, {})
        count = s.get("count", 0)
        since = s.get("first_seen", "today")
        mode  = user_modes.get(user_id, "normal").capitalize()
        bot.answer_callback_query(call.id)
        bot.send_message(
            call.message.chat.id,
            f"📊 <b>Stats:</b>\nMessages: <b>{count}</b>\nSince: <b>{since}</b>\nMode: <b>{mode}</b>",
            parse_mode="HTML",
        )


# ── Main Message Handler ──────────────────────────────────────────────────────
@bot.message_handler(func=lambda m: m.text and not m.text.startswith("/"))
def handle_message(message):
    user_id   = message.from_user.id
    chat_id   = message.chat.id
    user_text = message.text.strip()

    logger.info(f"User {user_id} [{user_modes.get(user_id,'normal')}]: {user_text[:80]}")
    update_stats(user_id)
    send_typing(chat_id)

    # Send reaction (raw API call — reliable)
    emoji = pick_reaction(user_text)
    send_reaction(chat_id, message.message_id, emoji)

    # Get AI reply
    raw_reply = get_ai_response(user_id, user_text)
    formatted = format_for_telegram(raw_reply)

    try:
        bot.reply_to(message, formatted, parse_mode="HTML")
    except Exception as e:
        logger.warning(f"HTML parse failed ({e}), sending plain text")
        bot.reply_to(message, raw_reply)


@bot.message_handler(content_types=["photo", "document", "sticker", "voice", "video"])
def handle_unsupported(message):
    bot.reply_to(
        message,
        "📝 I currently support <b>text messages</b> only. Please type your question!",
        parse_mode="HTML",
    )


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT       = int(os.environ.get("PORT", 5000))
    RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/{BOT_TOKEN}"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook set → {webhook_url}")
        app.run(host="0.0.0.0", port=PORT)
    else:
        logger.info("Starting in polling mode (local dev)...")
        bot.remove_webhook()
        threading.Thread(target=bot.infinity_polling, daemon=True).start()
        app.run(host="0.0.0.0", port=PORT)
