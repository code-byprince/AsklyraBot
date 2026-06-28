import os
import json
import time
import logging
import tempfile
import zipfile
import csv
import threading
import re
from datetime import datetime, timedelta
from collections import defaultdict
from io import StringIO, BytesIO

import telebot
import httpx
from telebot import types
from flask import Flask, request
from openai import OpenAI

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip().isdigit()]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")   # e.g. https://yourapp.onrender.com

MODEL = "deepseek-ai/DeepSeek-V3-0324:novita"
MAX_HISTORY = 30          # messages kept in memory per user
SUMMARY_EVERY = 20        # summarise after N messages
MAX_TOKENS = 1800
RATE_LIMIT_WINDOW = 60    # seconds
RATE_LIMIT_MAX = 25       # max messages per window

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CLIENTS
# ─────────────────────────────────────────────
bot    = telebot.TeleBot(BOT_TOKEN, parse_mode=None)
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN, http_client=httpx.Client())
app    = Flask(__name__)

# ─────────────────────────────────────────────
#  IN-MEMORY STORES
# ─────────────────────────────────────────────
user_histories:   dict[int, list]   = defaultdict(list)
user_summaries:   dict[int, str]    = {}
user_languages:   dict[int, str]    = {}
user_msg_counts:  dict[int, int]    = defaultdict(int)
user_last_window: dict[int, float]  = defaultdict(float)
user_tokens_used: dict[int, int]    = defaultdict(int)
user_first_seen:  dict[int, str]    = {}
user_last_active: dict[int, str]    = {}
daily_messages:   dict[str, int]    = defaultdict(int)
error_logs:       list              = []
response_times:   list              = []
partial_responses:dict[int, str]    = {}    # for "continue" button
group_admins:     dict[int, list]   = {}
user_names:       dict[int, str]    = {}

stats_lock = threading.Lock()

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
SUPPORTED_LANGS = {
    "Hindi": "hi", "English": "en", "Spanish": "es", "French": "fr",
    "German": "de", "Japanese": "ja", "Chinese": "zh", "Arabic": "ar",
    "Russian": "ru", "Portuguese": "pt", "Italian": "it", "Korean": "ko",
    "Bengali": "bn", "Urdu": "ur", "Turkish": "tr", "Vietnamese": "vi",
    "Thai": "th", "Dutch": "nl", "Polish": "pl", "Swedish": "sv",
    "Norwegian": "no", "Danish": "da", "Finnish": "fi", "Greek": "el",
    "Hebrew": "he", "Indonesian": "id", "Malay": "ms", "Filipino": "fil",
    "Swahili": "sw", "Persian": "fa", "Punjabi": "pa", "Gujarati": "gu",
    "Marathi": "mr", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
}

REACTIONS = ["👍", "🔥", "❤️", "🎉", "💡", "🧠", "✅", "🚀", "😊", "🤔"]

def get_reaction(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["error","wrong","sorry","fail","issue"]): return "😟"
    if any(w in t for w in ["code","python","function","program","script"]): return "💻"
    if any(w in t for w in ["math","calcul","equation","formula"]): return "🔢"
    if any(w in t for w in ["hello","hi","hey","namaste","hola"]): return "👋"
    if any(w in t for w in ["love","heart","romantic","beautiful"]): return "❤️"
    if any(w in t for w in ["news","politic","world","country"]): return "🌍"
    if any(w in t for w in ["food","recipe","cook","eat","dish"]): return "🍽️"
    if any(w in t for w in ["science","research","study","discover"]): return "🔬"
    if any(w in t for w in ["music","song","beat","melody"]): return "🎵"
    if any(w in t for w in ["sport","game","match","team","player"]): return "⚽"
    return "🤖"

def clean_for_telegram(text: str) -> str:
    """Remove markdown symbols (#, *, _) but keep bold via <b> tags."""
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Convert **bold** or __bold__ → <b>bold</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Remove remaining * or _
    text = re.sub(r"(?<!\w)[*_](?!\w)", "", text)
    # Remove ` backtick blocks (keep content)
    text = re.sub(r"```[\w]*\n?([\s\S]*?)```", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()

def split_long_message(text: str, limit: int = 3800) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        parts.append(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        parts.append(text)
    return parts

def detect_language(text: str) -> str:
    """Simple heuristic language detector using AI."""
    return "auto"

def record_error(err: str):
    with stats_lock:
        error_logs.append({"time": datetime.now().isoformat(), "error": err[-300:]})
        if len(error_logs) > 200:
            error_logs.pop(0)

def update_user_meta(uid: int, name: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    user_names[uid] = name
    if uid not in user_first_seen:
        user_first_seen[uid] = now
    user_last_active[uid] = now

def check_rate_limit(uid: int) -> bool:
    now = time.time()
    with stats_lock:
        if now - user_last_window[uid] > RATE_LIMIT_WINDOW:
            user_last_window[uid] = now
            user_msg_counts[uid] = 0
        user_msg_counts[uid] += 1
        return user_msg_counts[uid] <= RATE_LIMIT_MAX

def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def increment_daily(uid: int):
    with stats_lock:
        daily_messages[today()] += 1

# ─────────────────────────────────────────────
#  AI CORE
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an advanced, friendly AI assistant on Telegram.

RULES (STRICTLY FOLLOW):
1. Never use # symbols or markdown headings.
2. Never use * or _ for italic/bold — the system will handle bold automatically.
3. Use **double asterisks** around key points so they get displayed as bold.
4. Keep responses clear, structured, and conversational.
5. Auto-detect the user's language and always reply in the SAME language.
6. Be helpful, accurate, and concise. For long topics, split logically.
7. If asked to translate, translate accurately into the requested language.
8. Add relevant emojis to make replies engaging (but don't overdo it).
"""

def build_messages(uid: int, user_text: str) -> list:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if uid in user_summaries:
        msgs.append({"role": "system", "content": f"[Conversation summary so far]: {user_summaries[uid]}"})
    msgs.extend(user_histories[uid])
    msgs.append({"role": "user", "content": user_text})
    return msgs

def ask_ai(uid: int, user_text: str, extra_context: str = "") -> str:
    prompt = (extra_context + "\n\n" + user_text).strip() if extra_context else user_text
    msgs = build_messages(uid, prompt)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        answer = resp.choices[0].message.content or ""
        tokens = getattr(resp.usage, "total_tokens", 0)
        elapsed = round(time.time() - t0, 2)
        with stats_lock:
            user_tokens_used[uid] += tokens
            response_times.append(elapsed)
            if len(response_times) > 500:
                response_times.pop(0)
        # Update history
        user_histories[uid].append({"role": "user", "content": user_text})
        user_histories[uid].append({"role": "assistant", "content": answer})
        # Trim + summarise if needed
        if len(user_histories[uid]) >= SUMMARY_EVERY * 2:
            summarise_history(uid)
        elif len(user_histories[uid]) > MAX_HISTORY * 2:
            user_histories[uid] = user_histories[uid][-(MAX_HISTORY * 2):]
        return answer
    except Exception as e:
        record_error(str(e))
        log.error("AI error: %s", e)
        return "⚠️ Abhi AI se response nahi mila. Thodi der baad try karo."

def summarise_history(uid: int):
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in user_histories[uid]
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize the following conversation in 3-5 sentences, preserving key facts and context."},
                {"role": "user", "content": history_text}
            ],
            max_tokens=400,
        )
        summary = resp.choices[0].message.content or ""
        user_summaries[uid] = summary
        user_histories[uid] = []
        log.info("Summarised history for user %s", uid)
    except Exception as e:
        record_error(str(e))
        user_histories[uid] = user_histories[uid][-MAX_HISTORY:]

def continue_markup(uid: int) -> types.InlineKeyboardMarkup | None:
    if uid in partial_responses and partial_responses[uid]:
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton("▶️ Continue response", callback_data=f"continue_{uid}"))
        return kb
    return None

# ─────────────────────────────────────────────
#  DOCUMENT PROCESSORS
# ─────────────────────────────────────────────
def process_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text[:8000] or "PDF mein readable text nahi mila."
    except Exception as e:
        return f"PDF read error: {e}"

def process_docx(file_bytes: bytes) -> str:
    try:
        import docx
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)[:8000]
    except Exception as e:
        return f"DOCX read error: {e}"

def process_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")[:8000]
    except Exception as e:
        return f"TXT read error: {e}"

def process_csv(file_bytes: bytes) -> str:
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.reader(StringIO(text))
        rows = list(reader)[:50]
        return "\n".join([", ".join(r) for r in rows])
    except Exception as e:
        return f"CSV read error: {e}"

def process_excel(file_bytes: bytes) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(BytesIO(file_bytes), read_only=True)
        result = []
        for sheet in wb.worksheets[:3]:
            result.append(f"Sheet: {sheet.title}")
            for row in list(sheet.iter_rows(values_only=True))[:30]:
                result.append(", ".join(str(c) if c is not None else "" for c in row))
        return "\n".join(result)[:8000]
    except Exception as e:
        return f"Excel read error: {e}"

def process_json(file_bytes: bytes) -> str:
    try:
        data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        return json.dumps(data, indent=2, ensure_ascii=False)[:8000]
    except Exception as e:
        return f"JSON parse error: {e}"

def process_zip(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as z:
            names = z.namelist()
            summary = [f"ZIP contains {len(names)} files:"]
            summary.extend(names[:30])
            # Try reading text files inside
            for name in names[:5]:
                if name.endswith((".txt", ".py", ".js", ".md", ".csv", ".json")):
                    try:
                        content = z.read(name).decode("utf-8", errors="ignore")[:500]
                        summary.append(f"\n--- {name} ---\n{content}")
                    except Exception:
                        pass
            return "\n".join(summary)
    except Exception as e:
        return f"ZIP read error: {e}"

def process_code(file_bytes: bytes, ext: str) -> str:
    return file_bytes.decode("utf-8", errors="ignore")[:8000]

# ─────────────────────────────────────────────
#  MESSAGE SENDER
# ─────────────────────────────────────────────
def send_reply(message: types.Message, text: str, reaction: str = ""):
    uid = message.from_user.id
    cid = message.chat.id
    cleaned = clean_for_telegram(text)
    parts = split_long_message(cleaned)

    # React to message (first part only)
    if reaction:
        try:
            bot.set_message_reaction(cid, message.message_id, [types.ReactionTypeEmoji(reaction)])
        except Exception:
            pass

    for i, part in enumerate(parts):
        is_last = (i == len(parts) - 1)
        markup = None
        if is_last and len(parts) > 1:
            # Offer "continue" if response was truncated
            partial_responses[uid] = ""
        try:
            bot.send_message(cid, part, parse_mode="HTML", reply_markup=markup)
        except Exception as e:
            record_error(str(e))
            try:
                bot.send_message(cid, part)
            except Exception:
                pass

# ─────────────────────────────────────────────
#  COMMAND HANDLERS
# ─────────────────────────────────────────────
@bot.message_handler(commands=["start"])
def cmd_start(msg: types.Message):
    uid = msg.from_user.id
    name = msg.from_user.first_name or "Dost"
    update_user_meta(uid, name)
    text = (
        f"👋 <b>Hello {name}!</b>\n\n"
        "Main hun <b>DeepSeek AI Bot</b> — tera personal AI assistant! 🤖\n\n"
        "🔥 <b>Main kya kar sakta hun:</b>\n"
        "• Koi bhi sawaal ka jawab\n"
        "• Files analyse karna (PDF, DOCX, Excel, CSV, ZIP...)\n"
        "• Code likhna aur debug karna\n"
        "• 100+ languages mein translate karna\n"
        "• Voice messages samajhna\n"
        "• Group admin tools\n\n"
        "📌 <b>Commands:</b>\n"
        "/help — Saari commands\n"
        "/stats — Teri stats\n"
        "/translate — Translation\n"
        "/clear — Chat history clear\n"
        "/language — Language set karo\n"
        "/admin — Admin tools (admins only)\n\n"
        "💬 Bas message karo — main hoon! 🚀"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")
    try:
        bot.set_message_reaction(msg.chat.id, msg.message_id, [types.ReactionTypeEmoji("👋")])
    except Exception:
        pass

@bot.message_handler(commands=["help"])
def cmd_help(msg: types.Message):
    text = (
        "📚 <b>Complete Command List</b>\n\n"
        "🤖 <b>AI Features:</b>\n"
        "• Bas kuch bhi type karo — AI jawab dega\n"
        "• Files bhejo — analyse ho jaayenge\n"
        "• Voice bhejo — samjha jaayega\n\n"
        "⚙️ <b>Commands:</b>\n"
        "/start — Bot start\n"
        "/help — Ye message\n"
        "/clear — Apni chat history clear karo\n"
        "/stats — Apni personal stats\n"
        "/mystats — Token aur usage info\n"
        "/translate [lang] [text] — Translate karo\n"
        "/language — Preferred language set karo\n"
        "/summary — Conversation ka summary\n"
        "/leaderboard — Top users\n\n"
        "👑 <b>Admin Commands:</b>\n"
        "/admin — Admin panel\n"
        "/totalusers — Total users count\n"
        "/activesers — Active users today\n"
        "/dailymsgs — Aaj ke messages\n"
        "/errors — Recent errors\n"
        "/modelstats — AI model usage\n\n"
        "📎 <b>Supported Files:</b>\n"
        "PDF • DOCX • TXT • CSV • Excel • JSON • ZIP • Code files • Images\n\n"
        "🎙️ Voice messages bhi support hain!"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")

@bot.message_handler(commands=["clear"])
def cmd_clear(msg: types.Message):
    uid = msg.from_user.id
    user_histories[uid] = []
    user_summaries.pop(uid, None)
    bot.send_message(msg.chat.id, "🗑️ <b>Chat history clear ho gayi!</b> Ab fresh start karo.", parse_mode="HTML")

@bot.message_handler(commands=["stats", "mystats"])
def cmd_stats(msg: types.Message):
    uid = msg.from_user.id
    name = msg.from_user.first_name or "User"
    tokens = user_tokens_used.get(uid, 0)
    msgs_in_history = len(user_histories.get(uid, []))
    first = user_first_seen.get(uid, "N/A")
    last = user_last_active.get(uid, "N/A")
    summary = "Available ✅" if uid in user_summaries else "None"
    text = (
        f"📊 <b>{name} ki Stats</b>\n\n"
        f"🔑 Tokens used: <b>{tokens:,}</b>\n"
        f"💬 Messages in memory: <b>{msgs_in_history}</b>\n"
        f"📝 Long-term summary: {summary}\n"
        f"📅 First seen: <b>{first}</b>\n"
        f"🕐 Last active: <b>{last}</b>\n"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")

@bot.message_handler(commands=["summary"])
def cmd_summary(msg: types.Message):
    uid = msg.from_user.id
    if uid in user_summaries:
        bot.send_message(msg.chat.id, f"📝 <b>Conversation Summary:</b>\n\n{user_summaries[uid]}", parse_mode="HTML")
    elif user_histories.get(uid):
        bot.send_message(msg.chat.id, "⏳ Summary ban rahi hai...", parse_mode="HTML")
        summarise_history(uid)
        if uid in user_summaries:
            bot.send_message(msg.chat.id, f"📝 <b>Summary:</b>\n\n{user_summaries[uid]}", parse_mode="HTML")
    else:
        bot.send_message(msg.chat.id, "ℹ️ Abhi koi conversation history nahi hai.", parse_mode="HTML")

@bot.message_handler(commands=["translate"])
def cmd_translate(msg: types.Message):
    uid = msg.from_user.id
    parts = msg.text.split(None, 2)
    if len(parts) < 3:
        # Show language keyboard
        kb = types.InlineKeyboardMarkup(row_width=3)
        buttons = [types.InlineKeyboardButton(lang, callback_data=f"setlang_{code}")
                   for lang, code in list(SUPPORTED_LANGS.items())[:30]]
        kb.add(*buttons)
        bot.send_message(msg.chat.id,
            "🌐 <b>Translation</b>\n\nUsage: /translate [language] [text]\n\nExample:\n/translate Hindi Hello, how are you?\n\nYa apni language set karo:",
            parse_mode="HTML", reply_markup=kb)
        return
    lang = parts[1]
    text_to_translate = parts[2]
    update_user_meta(uid, msg.from_user.first_name or "")
    ai_prompt = f"Translate the following text to {lang}. Only output the translation, nothing else:\n\n{text_to_translate}"
    loading = bot.send_message(msg.chat.id, "🌐 Translating...")
    result = ask_ai(uid, ai_prompt)
    bot.delete_message(msg.chat.id, loading.message_id)
    bot.send_message(msg.chat.id, f"🌐 <b>Translation ({lang}):</b>\n\n{clean_for_telegram(result)}", parse_mode="HTML")

@bot.message_handler(commands=["language"])
def cmd_language(msg: types.Message):
    kb = types.InlineKeyboardMarkup(row_width=3)
    buttons = [types.InlineKeyboardButton(lang, callback_data=f"setlang_{code}")
               for lang, code in list(SUPPORTED_LANGS.items())[:24]]
    kb.add(*buttons)
    bot.send_message(msg.chat.id, "🌐 <b>Apni preferred language choose karo:</b>", parse_mode="HTML", reply_markup=kb)

@bot.message_handler(commands=["leaderboard"])
def cmd_leaderboard(msg: types.Message):
    sorted_users = sorted(user_tokens_used.items(), key=lambda x: x[1], reverse=True)[:10]
    if not sorted_users:
        bot.send_message(msg.chat.id, "📊 Abhi koi data nahi hai.", parse_mode="HTML")
        return
    lines = ["🏆 <b>User Leaderboard (by tokens used)</b>\n"]
    medals = ["🥇", "🥈", "🥉"] + ["🔹"] * 7
    for i, (uid, tokens) in enumerate(sorted_users):
        name = user_names.get(uid, f"User {uid}")
        lines.append(f"{medals[i]} <b>{name}</b> — {tokens:,} tokens")
    bot.send_message(msg.chat.id, "\n".join(lines), parse_mode="HTML")

# ─────────────────────────────────────────────
#  ADMIN COMMANDS
# ─────────────────────────────────────────────
def is_admin(uid: int) -> bool:
    return uid in ADMIN_IDS

@bot.message_handler(commands=["admin"])
def cmd_admin(msg: types.Message):
    if not is_admin(msg.from_user.id):
        bot.send_message(msg.chat.id, "🚫 Tujhe access nahi hai.")
        return
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("👥 Total Users", callback_data="adm_totalusers"),
        types.InlineKeyboardButton("🟢 Active Today", callback_data="adm_active"),
        types.InlineKeyboardButton("📨 Daily Msgs", callback_data="adm_daily"),
        types.InlineKeyboardButton("⚠️ Error Logs", callback_data="adm_errors"),
        types.InlineKeyboardButton("⏱ Response Time", callback_data="adm_resptime"),
        types.InlineKeyboardButton("🤖 Model Usage", callback_data="adm_model"),
    )
    bot.send_message(msg.chat.id, "👑 <b>Admin Panel</b>", parse_mode="HTML", reply_markup=kb)

@bot.message_handler(commands=["totalusers"])
def cmd_totalusers(msg: types.Message):
    if not is_admin(msg.from_user.id): return
    bot.send_message(msg.chat.id, f"👥 <b>Total Users:</b> {len(user_first_seen)}", parse_mode="HTML")

@bot.message_handler(commands=["activeusers"])
def cmd_activeusers(msg: types.Message):
    if not is_admin(msg.from_user.id): return
    today_str = today()
    active = sum(1 for dt in user_last_active.values() if dt.startswith(today_str))
    bot.send_message(msg.chat.id, f"🟢 <b>Active Users Today:</b> {active}", parse_mode="HTML")

@bot.message_handler(commands=["dailymsgs"])
def cmd_dailymsgs(msg: types.Message):
    if not is_admin(msg.from_user.id): return
    count = daily_messages.get(today(), 0)
    bot.send_message(msg.chat.id, f"📨 <b>Today's Messages:</b> {count}", parse_mode="HTML")

@bot.message_handler(commands=["errors"])
def cmd_errors(msg: types.Message):
    if not is_admin(msg.from_user.id): return
    if not error_logs:
        bot.send_message(msg.chat.id, "✅ No errors logged.", parse_mode="HTML")
        return
    lines = [f"• [{e['time']}] {e['error']}" for e in error_logs[-10:]]
    bot.send_message(msg.chat.id, "⚠️ <b>Recent Errors:</b>\n\n" + "\n".join(lines), parse_mode="HTML")

@bot.message_handler(commands=["modelstats"])
def cmd_modelstats(msg: types.Message):
    if not is_admin(msg.from_user.id): return
    total_tokens = sum(user_tokens_used.values())
    avg_rt = round(sum(response_times) / len(response_times), 2) if response_times else 0
    text = (
        f"🤖 <b>Model Stats</b>\n\n"
        f"Model: <b>{MODEL}</b>\n"
        f"Total tokens used: <b>{total_tokens:,}</b>\n"
        f"Avg response time: <b>{avg_rt}s</b>\n"
        f"Total users: <b>{len(user_first_seen)}</b>\n"
        f"Error count: <b>{len(error_logs)}</b>\n"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")

# ─────────────────────────────────────────────
#  CALLBACK QUERY HANDLER
# ─────────────────────────────────────────────
@bot.callback_query_handler(func=lambda c: True)
def handle_callback(call: types.CallbackQuery):
    uid = call.from_user.id
    data = call.data

    if data.startswith("setlang_"):
        lang_code = data.replace("setlang_", "")
        user_languages[uid] = lang_code
        lang_name = next((k for k, v in SUPPORTED_LANGS.items() if v == lang_code), lang_code)
        bot.answer_callback_query(call.id, f"✅ Language set to {lang_name}")
        bot.edit_message_text(f"✅ <b>Language set to {lang_name}!</b>\nAb main {lang_name} mein jawab dunga.", 
                              call.message.chat.id, call.message.message_id, parse_mode="HTML")

    elif data.startswith("continue_"):
        stored = partial_responses.get(uid, "")
        if stored:
            bot.answer_callback_query(call.id, "▶️ Continuing...")
            result = ask_ai(uid, "Please continue your previous response from where you left off.")
            cleaned = clean_for_telegram(result)
            parts = split_long_message(cleaned)
            for part in parts:
                bot.send_message(call.message.chat.id, part, parse_mode="HTML")
        else:
            bot.answer_callback_query(call.id, "Nothing to continue.")

    elif data == "adm_totalusers":
        if is_admin(uid):
            bot.answer_callback_query(call.id, f"Total Users: {len(user_first_seen)}")

    elif data == "adm_active":
        if is_admin(uid):
            today_str = today()
            active = sum(1 for dt in user_last_active.values() if dt.startswith(today_str))
            bot.answer_callback_query(call.id, f"Active Today: {active}")

    elif data == "adm_daily":
        if is_admin(uid):
            count = daily_messages.get(today(), 0)
            bot.answer_callback_query(call.id, f"Today's Messages: {count}")

    elif data == "adm_errors":
        if is_admin(uid):
            bot.answer_callback_query(call.id, f"Errors: {len(error_logs)}")

    elif data == "adm_resptime":
        if is_admin(uid):
            avg = round(sum(response_times)/len(response_times), 2) if response_times else 0
            bot.answer_callback_query(call.id, f"Avg Response: {avg}s")

    elif data == "adm_model":
        if is_admin(uid):
            total = sum(user_tokens_used.values())
            bot.answer_callback_query(call.id, f"Total Tokens: {total:,}")

# ─────────────────────────────────────────────
#  DOCUMENT HANDLER
# ─────────────────────────────────────────────
@bot.message_handler(content_types=["document"])
def handle_document(msg: types.Message):
    uid = msg.from_user.id
    update_user_meta(uid, msg.from_user.first_name or "")
    increment_daily(uid)

    if not check_rate_limit(uid):
        bot.send_message(msg.chat.id, "⚠️ Thoda slow down karo! 1 minute baad try karo.")
        return

    doc = msg.document
    fname = doc.file_name or "file"
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""

    loading = bot.send_message(msg.chat.id, f"📂 <b>{fname}</b> process ho rahi hai...", parse_mode="HTML")
    try:
        file_info = bot.get_file(doc.file_id)
        file_bytes = bot.download_file(file_info.file_path)

        if ext == "pdf":
            content = process_pdf(file_bytes)
            file_type = "PDF"
        elif ext == "docx":
            content = process_docx(file_bytes)
            file_type = "Word Document"
        elif ext == "txt":
            content = process_txt(file_bytes)
            file_type = "Text File"
        elif ext == "csv":
            content = process_csv(file_bytes)
            file_type = "CSV File"
        elif ext in ("xlsx", "xls"):
            content = process_excel(file_bytes)
            file_type = "Excel File"
        elif ext == "json":
            content = process_json(file_bytes)
            file_type = "JSON File"
        elif ext == "zip":
            content = process_zip(file_bytes)
            file_type = "ZIP Archive"
        elif ext in ("py","js","ts","java","cpp","c","cs","go","rb","php","html","css","sql","sh","yaml","yml","xml","md"):
            content = process_code(file_bytes, ext)
            file_type = f"Code File (.{ext})"
        else:
            content = process_txt(file_bytes)
            file_type = "File"

        caption = msg.caption or ""
        user_question = caption if caption else f"Is {file_type} ka content analyse karo aur key points batao."
        extra_ctx = f"[File: {fname} | Type: {file_type}]\n\nContent:\n{content}"

        bot.delete_message(msg.chat.id, loading.message_id)
        result = ask_ai(uid, user_question, extra_ctx)
        reaction = get_reaction(result)
        send_reply(msg, result, reaction)

    except Exception as e:
        record_error(str(e))
        log.error("Doc error: %s", e)
        bot.delete_message(msg.chat.id, loading.message_id)
        bot.send_message(msg.chat.id, f"❌ File process nahi ho saki: {str(e)[:200]}", parse_mode="HTML")

# ─────────────────────────────────────────────
#  VOICE HANDLER
# ─────────────────────────────────────────────
@bot.message_handler(content_types=["voice"])
def handle_voice(msg: types.Message):
    uid = msg.from_user.id
    update_user_meta(uid, msg.from_user.first_name or "")
    increment_daily(uid)

    if not check_rate_limit(uid):
        bot.send_message(msg.chat.id, "⚠️ Thoda slow down karo!")
        return

    loading = bot.send_message(msg.chat.id, "🎙️ Voice message sun raha hun...", parse_mode="HTML")
    try:
        file_info = bot.get_file(msg.voice.file_id)
        audio_bytes = bot.download_file(file_info.file_path)

        # Try transcription via OpenAI-compatible Whisper if available
        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            with open(tmp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="openai/whisper-large-v3",
                    file=audio_file,
                )
            transcribed_text = transcript.text
            os.unlink(tmp_path)
        except Exception:
            # Fallback: inform user
            bot.delete_message(msg.chat.id, loading.message_id)
            bot.send_message(msg.chat.id,
                "🎙️ Voice message mila! Lekin automatic transcription ke liye please text mein likhkar bhejo.\n\nVoice support: Aapka message receive ho gaya.",
                parse_mode="HTML")
            return

        bot.delete_message(msg.chat.id, loading.message_id)
        bot.send_message(msg.chat.id, f"🎙️ <b>Suna:</b> {transcribed_text}", parse_mode="HTML")
        result = ask_ai(uid, transcribed_text)
        send_reply(msg, result, get_reaction(result))

    except Exception as e:
        record_error(str(e))
        bot.delete_message(msg.chat.id, loading.message_id)
        bot.send_message(msg.chat.id, "❌ Voice process nahi ho saki.", parse_mode="HTML")

# ─────────────────────────────────────────────
#  PHOTO HANDLER
# ─────────────────────────────────────────────
@bot.message_handler(content_types=["photo"])
def handle_photo(msg: types.Message):
    uid = msg.from_user.id
    update_user_meta(uid, msg.from_user.first_name or "")
    increment_daily(uid)
    caption = msg.caption or "Is image mein kya hai? Describe karo."
    loading = bot.send_message(msg.chat.id, "🖼️ Image dekh raha hun...", parse_mode="HTML")
    try:
        photo = msg.photo[-1]
        file_info = bot.get_file(photo.file_id)
        img_bytes = bot.download_file(file_info.file_path)
        import base64
        b64 = base64.b64encode(img_bytes).decode()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": caption}
                ]}
            ],
            max_tokens=MAX_TOKENS,
        )
        answer = resp.choices[0].message.content or "Image samajh nahi aaya."
        bot.delete_message(msg.chat.id, loading.message_id)
        send_reply(msg, answer, "🖼️")
    except Exception as e:
        record_error(str(e))
        bot.delete_message(msg.chat.id, loading.message_id)
        # Fallback: just ask about it textually
        result = ask_ai(uid, caption + " (Note: Image provided but vision not available in current model)")
        send_reply(msg, result, "🖼️")

# ─────────────────────────────────────────────
#  GROUP ADMIN TOOLS
# ─────────────────────────────────────────────
@bot.message_handler(commands=["kick", "ban", "mute", "unmute", "warn"])
def cmd_group_admin(msg: types.Message):
    cid = msg.chat.id
    uid = msg.from_user.id
    if msg.chat.type not in ["group", "supergroup"]:
        bot.send_message(cid, "ℹ️ Ye command sirf groups mein kaam karti hai.")
        return
    try:
        admins = bot.get_chat_administrators(cid)
        admin_ids = [a.user.id for a in admins]
        if uid not in admin_ids:
            bot.send_message(cid, "🚫 Sirf admins ye command use kar sakte hain.")
            return
        if not msg.reply_to_message:
            bot.send_message(cid, "ℹ️ Kisi message ko reply karo.", parse_mode="HTML")
            return
        target = msg.reply_to_message.from_user
        cmd = msg.text.split()[0][1:].lower()
        if cmd == "kick":
            bot.ban_chat_member(cid, target.id)
            bot.unban_chat_member(cid, target.id)
            bot.send_message(cid, f"👢 <b>{target.first_name}</b> ko kick kar diya.", parse_mode="HTML")
        elif cmd == "ban":
            bot.ban_chat_member(cid, target.id)
            bot.send_message(cid, f"🔨 <b>{target.first_name}</b> ban ho gaya.", parse_mode="HTML")
        elif cmd == "mute":
            bot.restrict_chat_member(cid, target.id, types.ChatPermissions(can_send_messages=False))
            bot.send_message(cid, f"🔇 <b>{target.first_name}</b> mute ho gaya.", parse_mode="HTML")
        elif cmd == "unmute":
            bot.restrict_chat_member(cid, target.id, types.ChatPermissions(can_send_messages=True))
            bot.send_message(cid, f"🔊 <b>{target.first_name}</b> unmute ho gaya.", parse_mode="HTML")
        elif cmd == "warn":
            bot.send_message(cid, f"⚠️ <b>{target.first_name}</b> ko warning mili! Behaviour theek karo.", parse_mode="HTML")
    except Exception as e:
        record_error(str(e))
        bot.send_message(cid, f"❌ Action fail: {str(e)[:100]}", parse_mode="HTML")

# ─────────────────────────────────────────────
#  MAIN TEXT HANDLER
# ─────────────────────────────────────────────
@bot.message_handler(func=lambda m: m.text and not m.text.startswith("/"))
def handle_text(msg: types.Message):
    uid  = msg.from_user.id
    name = msg.from_user.first_name or "User"
    text = msg.text.strip()

    update_user_meta(uid, name)
    increment_daily(uid)

    if not check_rate_limit(uid):
        bot.send_message(msg.chat.id,
            "⏳ <b>Thoda ruk ja bhai!</b>\nTu bahut tez chal raha hai 😅\n1 minute baad phir try karo.",
            parse_mode="HTML")
        return

    # Typing indicator
    bot.send_chat_action(msg.chat.id, "typing")

    # Language preference in prompt
    lang_pref = user_languages.get(uid, "")
    extra = f"[User's preferred language: {lang_pref}. Reply in that language.]\n" if lang_pref else ""

    loading = bot.send_message(msg.chat.id, "🧠 Soch raha hun...", parse_mode="HTML")
    result = ask_ai(uid, text, extra)
    try:
        bot.delete_message(msg.chat.id, loading.message_id)
    except Exception:
        pass

    reaction = get_reaction(text + " " + result)
    send_reply(msg, result, reaction)

# ─────────────────────────────────────────────
#  FLASK WEBHOOK
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return "🤖 DeepSeek Telegram Bot is running!", 200

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        json_str = request.get_data(as_text=True)
        update = types.Update.de_json(json_str)
        bot.process_new_updates([update])
    return "OK", 200

@app.route("/health", methods=["GET"])
def health():
    return json.dumps({
        "status": "ok",
        "users": len(user_first_seen),
        "errors": len(error_logs),
        "model": MODEL,
    }), 200

# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────
def setup_webhook():
    if WEBHOOK_URL:
        webhook_url = f"{WEBHOOK_URL.rstrip('/')}/{BOT_TOKEN}"
        bot.remove_webhook()
        time.sleep(1)
        bot.set_webhook(url=webhook_url)
        log.info("Webhook set to: %s", webhook_url)
    else:
        log.warning("WEBHOOK_URL not set — running in polling mode for local testing.")

if __name__ == "__main__":
    if WEBHOOK_URL:
        setup_webhook()
        port = int(os.environ.get("PORT", 10000))
        log.info("Starting Flask on port %s", port)
        app.run(host="0.0.0.0", port=port)
    else:
        log.info("Starting polling mode...")
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
