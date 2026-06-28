# -*- coding: utf-8 -*-
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
            "\n<b>🔐 Admin Commands:</b>\n"
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


@bot.message_handler(commands=["clear"])
def cmd_clear(msg):
    user_histories[msg.from_user.id] = []
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
    reply = get_ai_response(msg.from_user.id, q)
    safe_send(msg.chat.id, format_for_telegram(reply), reply_to=msg.message_id)


@bot.message_handler(commands=["translate"])
def cmd_translate(msg):
    txt = msg.text.partition(" ")[2].strip()
    if not txt:
        bot.send_message(msg.chat.id, "Usage: /translate Aap kaise hain?")
        return
    track_message(msg.from_user.id)
    send_typing(msg.chat.id)
    send_reaction(msg.chat.id, msg.message_id, "🌐")
    reply = get_ai_response(msg.from_user.id,
        f"Detect the language and translate this to English:\n\n{txt}")
    safe_send(msg.chat.id, format_for_telegram(reply), reply_to=msg.message_id)


@bot.message_handler(commands=["summarize"])
def cmd_summarize(msg):
    txt = msg.text.partition(" ")[2].strip()
    if not txt:
        bot.send_message(msg.chat.id, "Usage: /summarize [long text here]")
        return
    track_message(msg.from_user.id)
    send_typing(msg.chat.id)
    send_reaction(msg.chat.id, msg.message_id, "📝")
    reply = get_ai_response(msg.from_user.id,
        f"Summarize in 3-5 bullet points:\n\n{txt}")
    safe_send(msg.chat.id, format_for_telegram(reply), reply_to=msg.message_id)


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
        user_histories[uid] = []
        bot.answer_callback_query(call.id, "History cleared!")
        safe_send(call.message.chat.id, "🗑️ <b>History cleared!</b> Fresh start.")

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
            f"• Model: <b>{MODEL}</b>"
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

    reply = get_ai_response(uid, prompt,
        extra_system=f"The user has uploaded a {label}. Analyze it thoroughly. Format output clearly with sections and bold key points.")
    bot.delete_message(chat_id, wait_msg.message_id)
    safe_send(chat_id, format_for_telegram(reply), reply_to=msg.message_id)


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

        reply = get_ai_response(uid, transcribed)
        safe_send(chat_id, format_for_telegram(reply), reply_to=msg.message_id)
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

    # Store name
    if uid not in user_stats:
        user_stats[uid] = {"count": 0, "first_seen": datetime.now().strftime("%Y-%m-%d"),
                           "name": "", "tokens_used": 0}
    user_stats[uid]["name"] = msg.from_user.first_name or ""

    track_message(uid)
    logger.info(f"User {uid} [{user_modes.get(uid,'normal')}]: {user_text[:60]}")

    send_typing(chat_id)
    send_reaction(chat_id, msg.message_id, pick_reaction(user_text))

    # Retry logic — up to 3 attempts on failure
    for attempt in range(3):
        raw_reply = get_ai_response(uid, user_text)
        if not raw_reply.startswith("⚠️"):
            break
        if attempt < 2:
            time.sleep(1.5)

    formatted = format_for_telegram(raw_reply)
    safe_send(chat_id, formatted, reply_to=msg.message_id)


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
