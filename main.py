import os
import json
import time
import logging
import tempfile
import zipfile
import csv
import threading
import re
import base64
from datetime import datetime
from collections import defaultdict
from io import StringIO, BytesIO

import httpx
import telebot
from telebot import types
from flask import Flask, request
from openai import OpenAI

# ══════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════
BOT_TOKEN      = os.environ.get("BOT_TOKEN", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
ADMIN_IDS      = [int(x) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip().isdigit()]
WEBHOOK_URL    = os.environ.get("WEBHOOK_URL", "")

MODEL          = "deepseek/deepseek-chat:free"
MAX_HISTORY    = 40
SUMMARY_EVERY  = 20
MAX_TOKENS     = 2000
RATE_LIMIT_WIN = 60
RATE_LIMIT_MAX = 100   # very generous limit

# ══════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
#  CLIENTS
# ══════════════════════════════════════════════════════
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    http_client=httpx.Client(timeout=60.0),
)
app = Flask(__name__)

# ══════════════════════════════════════════════════════
#  IN-MEMORY STORES
# ══════════════════════════════════════════════════════
user_histories:   dict = defaultdict(list)
user_summaries:   dict = {}
user_languages:   dict = {}
user_msg_counts:  dict = defaultdict(int)
user_last_window: dict = defaultdict(float)
user_tokens_used: dict = defaultdict(int)
user_first_seen:  dict = {}
user_last_active: dict = {}
daily_messages:   dict = defaultdict(int)
error_logs:       list = []
response_times:   list = []
user_names:       dict = {}
stats_lock = threading.Lock()

# ══════════════════════════════════════════════════════
#  SUPPORTED LANGUAGES
# ══════════════════════════════════════════════════════
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
    "Nepali": "ne", "Sinhala": "si", "Burmese": "my", "Khmer": "km",
    "Lao": "lo", "Mongolian": "mn", "Kazakh": "kk", "Uzbek": "uz",
}

# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def get_reaction(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["error", "wrong", "sorry", "fail", "issue", "galat"]): return "😟"
    if any(w in t for w in ["code", "python", "function", "program", "script", "bug"]): return "💻"
    if any(w in t for w in ["math", "calcul", "equation", "formula", "number"]): return "🔢"
    if any(w in t for w in ["hello", "hi", "hey", "namaste", "hola", "salam"]): return "👋"
    if any(w in t for w in ["love", "heart", "romantic", "beautiful", "pyaar"]): return "❤️"
    if any(w in t for w in ["news", "politic", "world", "country", "duniya"]): return "🌍"
    if any(w in t for w in ["food", "recipe", "cook", "eat", "khana", "dish"]): return "🍽️"
    if any(w in t for w in ["science", "research", "study", "discover", "vigyan"]): return "🔬"
    if any(w in t for w in ["music", "song", "beat", "melody", "gaana"]): return "🎵"
    if any(w in t for w in ["sport", "game", "match", "team", "khel"]): return "⚽"
    if any(w in t for w in ["money", "finance", "invest", "paisa", "profit"]): return "💰"
    if any(w in t for w in ["health", "doctor", "medicine", "sehat", "bimari"]): return "🏥"
    if any(w in t for w in ["travel", "trip", "tour", "safar", "yatra"]): return "✈️"
    if any(w in t for w in ["idea", "creative", "design", "art", "banao"]): return "🎨"
    return "🤖"

def clean_for_telegram(text: str) -> str:
    """Convert markdown to clean HTML for Telegram."""
    # Remove heading markers (#)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Convert **bold** or __bold__ → <b>bold</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text, flags=re.DOTALL)
    # Remove remaining lone * or _
    text = re.sub(r"(?<!\w)\*(?!\w)", "", text)
    text = re.sub(r"(?<!\w)_(?!\w)", "", text)
    # Convert ```code``` → plain text (keep content)
    text = re.sub(r"```[\w]*\n?([\s\S]*?)```", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove leftover # symbols
    text = text.replace("# ", "").replace("#", "")
    return text.strip()

def split_long_message(text: str, limit: int = 3800) -> list:
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

def record_error(err: str):
    with stats_lock:
        error_logs.append({"time": datetime.now().isoformat(), "error": str(err)[-400:]})
        if len(error_logs) > 200:
            error_logs.pop(0)

def update_user_meta(uid: int, name: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    user_names[uid] = name or f"User{uid}"
    if uid not in user_first_seen:
        user_first_seen[uid] = now
    user_last_active[uid] = now

def check_rate_limit(uid: int) -> bool:
    now = time.time()
    with stats_lock:
        if now - user_last_window[uid] > RATE_LIMIT_WIN:
            user_last_window[uid] = now
            user_msg_counts[uid] = 0
        user_msg_counts[uid] += 1
        return user_msg_counts[uid] <= RATE_LIMIT_MAX

def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def increment_daily():
    with stats_lock:
        daily_messages[today()] += 1

def send_typing(chat_id):
    try:
        bot.send_chat_action(chat_id, "typing")
    except Exception:
        pass

# ══════════════════════════════════════════════════════
#  AI CORE
# ══════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an advanced, friendly, and highly capable AI assistant on Telegram.

STRICT FORMATTING RULES:
1. NEVER use # symbols or markdown headings.
2. NEVER use * for bullet points or _ for italic.
3. Use **double asterisks** around key points so they appear bold.
4. Use plain text with emojis for structure instead of markdown symbols.
5. Keep responses clear, well-organized, and conversational.
6. Auto-detect the user's language and ALWAYS reply in the SAME language they use.
7. Add relevant emojis to make replies engaging but not excessive.
8. For lists, use • or numbers like 1. 2. 3. instead of * or -.
9. Be accurate, helpful, and thorough in your answers.
10. If the topic is complex, break it into clear sections using emojis as headers.
"""

def build_messages(uid: int, user_text: str) -> list:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if uid in user_summaries:
        msgs.append({
            "role": "system",
            "content": f"[Previous conversation summary]: {user_summaries[uid]}"
        })
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
            extra_headers={
                "HTTP-Referer": "https://t.me/",
                "X-Title": "Telegram AI Bot",
            },
        )
        answer = resp.choices[0].message.content or ""
        tokens = getattr(resp.usage, "total_tokens", 0) if resp.usage else 0
        elapsed = round(time.time() - t0, 2)

        with stats_lock:
            user_tokens_used[uid] += tokens
            response_times.append(elapsed)
            if len(response_times) > 500:
                response_times.pop(0)

        # Update history
        user_histories[uid].append({"role": "user", "content": user_text})
        user_histories[uid].append({"role": "assistant", "content": answer})

        # Summarise if history too long
        if len(user_histories[uid]) >= SUMMARY_EVERY * 2:
            threading.Thread(target=summarise_history, args=(uid,), daemon=True).start()
        elif len(user_histories[uid]) > MAX_HISTORY * 2:
            user_histories[uid] = user_histories[uid][-(MAX_HISTORY * 2):]

        return answer

    except Exception as e:
        record_error(str(e))
        log.error("AI error: %s", e)
        return "⚠️ AI se response nahi mila abhi. Thodi der baad try karo."

def summarise_history(uid: int):
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}" for m in user_histories[uid]
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize this conversation in 4-5 sentences. Keep all important facts, names, and context."},
                {"role": "user", "content": history_text}
            ],
            max_tokens=500,
            extra_headers={"HTTP-Referer": "https://t.me/", "X-Title": "Telegram AI Bot"},
        )
        summary = resp.choices[0].message.content or ""
        user_summaries[uid] = summary
        user_histories[uid] = []
        log.info("History summarised for user %s", uid)
    except Exception as e:
        record_error(str(e))
        user_histories[uid] = user_histories[uid][-MAX_HISTORY:]

# ══════════════════════════════════════════════════════
#  MESSAGE SENDER
# ══════════════════════════════════════════════════════
def send_reply(message: types.Message, text: str, reaction: str = ""):
    cid = message.chat.id
    mid = message.message_id
    cleaned = clean_for_telegram(text)
    parts = split_long_message(cleaned)

    # Add reaction
    if reaction:
        try:
            bot.set_message_reaction(cid, mid, [types.ReactionTypeEmoji(reaction)])
        except Exception:
            pass

    for i, part in enumerate(parts):
        try:
            bot.send_message(cid, part, parse_mode="HTML")
        except Exception as e:
            record_error(str(e))
            try:
                # Fallback: send as plain text
                bot.send_message(cid, re.sub(r"<[^>]+>", "", part))
            except Exception:
                pass

# ══════════════════════════════════════════════════════
#  FILE PROCESSORS
# ══════════════════════════════════════════════════════
def process_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text[:10000] or "PDF mein readable text nahi mila."
    except Exception as e:
        return f"PDF read error: {e}"

def process_docx(file_bytes: bytes) -> str:
    try:
        import docx
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())[:10000]
    except Exception as e:
        return f"DOCX read error: {e}"

def process_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")[:10000]

def process_csv(file_bytes: bytes) -> str:
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.reader(StringIO(text))
        rows = list(reader)[:60]
        return "\n".join([", ".join(str(c) for c in r) for r in rows])
    except Exception as e:
        return f"CSV read error: {e}"

def process_excel(file_bytes: bytes) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(BytesIO(file_bytes), read_only=True)
        result = []
        for sheet in wb.worksheets[:3]:
            result.append(f"Sheet: {sheet.title}")
            for row in list(sheet.iter_rows(values_only=True))[:40]:
                result.append(", ".join(str(c) if c is not None else "" for c in row))
        return "\n".join(result)[:10000]
    except Exception as e:
        return f"Excel read error: {e}"

def process_json(file_bytes: bytes) -> str:
    try:
        data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        return json.dumps(data, indent=2, ensure_ascii=False)[:10000]
    except Exception as e:
        return f"JSON parse error: {e}"

def process_zip(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as z:
            names = z.namelist()
            summary = [f"ZIP mein {len(names)} files hain:"]
            summary.extend(f"  • {n}" for n in names[:40])
            for name in names[:5]:
                if name.endswith((".txt", ".py", ".js", ".md", ".csv", ".json", ".html", ".css")):
                    try:
                        content = z.read(name).decode("utf-8", errors="ignore")[:600]
                        summary.append(f"\n--- {name} ---\n{content}")
                    except Exception:
                        pass
            return "\n".join(summary)
    except Exception as e:
        return f"ZIP read error: {e}"

def process_code(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")[:10000]

# ══════════════════════════════════════════════════════
#  COMMAND HANDLERS
# ══════════════════════════════════════════════════════
@bot.message_handler(commands=["start"])
def cmd_start(msg: types.Message):
    uid = msg.from_user.id
    name = msg.from_user.first_name or "Dost"
    update_user_meta(uid, name)
    text = (
        f"👋 <b>Hello {name}!</b>\n\n"
        "Main hun <b>LyraBot</b> — tera personal AI assistant! 🤖\n\n"
        "🔥 <b>Main kya kar sakta hun:</b>\n"
        "• Koi bhi sawaal ka jawab dunga\n"
        "• Files analyse karta hun (PDF, DOCX, Excel, CSV, ZIP...)\n"
        "• 40+ languages mein baat kar sakta hun\n"
        "• Code likhna, debug karna, explain karna\n"
        "• Voice messages samajhna\n"
        "• Teri poori conversation yaad rakhta hun\n\n"
        "📌 <b>Commands:</b>\n"
        "/help — Saari commands dekho\n"
        "/clear — Chat history clear karo\n"
        "/stats — Apni stats dekho\n"
        "/translate — Kuch bhi translate karo\n"
        "/summary — Conversation ka summary\n"
        "/leaderboard — Top users\n\n"
        "💬 <b>Bas kuch bhi type karo — main yahan hun!</b> 🚀"
    )
    try:
        bot.set_message_reaction(msg.chat.id, msg.message_id, [types.ReactionTypeEmoji("👋")])
    except Exception:
        pass
    bot.send_message(msg.chat.id, text, parse_mode="HTML")

@bot.message_handler(commands=["help"])
def cmd_help(msg: types.Message):
    text = (
        "📚 <b>Complete Command List</b>\n\n"
        "🤖 <b>AI Features:</b>\n"
        "• Kuch bhi type karo — AI jawab dega\n"
        "• Files bhejo — automatically analyse hogi\n"
        "• Voice message bhejo — samjha jaayega\n"
        "• Long-term memory — teri baatein yaad rahegi\n\n"
        "⚙️ <b>Commands:</b>\n"
        "/start — Bot start\n"
        "/help — Ye list\n"
        "/clear — Chat history clear\n"
        "/stats — Teri personal stats\n"
        "/summary — Is conversation ka summary\n"
        "/translate [language] [text] — Translate karo\n"
        "/language — Preferred language set karo\n"
        "/leaderboard — Top users list\n\n"
        "👑 <b>Admin Commands:</b>\n"
        "/admin — Admin panel\n"
        "/totalusers — Total users\n"
        "/activeusers — Aaj ke active users\n"
        "/dailymsgs — Aaj ke messages count\n"
        "/errors — Recent error logs\n"
        "/modelstats — AI usage stats\n\n"
        "👮 <b>Group Admin Tools:</b>\n"
        "/kick /ban /mute /unmute /warn — Member management\n\n"
        "📎 <b>Supported Files:</b>\n"
        "PDF • DOCX • TXT • CSV • Excel • JSON • ZIP • Code files\n\n"
        "🌐 <b>40+ Languages supported!</b>"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")

@bot.message_handler(commands=["clear"])
def cmd_clear(msg: types.Message):
    uid = msg.from_user.id
    user_histories[uid] = []
    user_summaries.pop(uid, None)
    try:
        bot.set_message_reaction(msg.chat.id, msg.message_id, [types.ReactionTypeEmoji("✅")])
    except Exception:
        pass
    bot.send_message(msg.chat.id, "🗑️ <b>Chat history clear ho gayi!</b>\n\nAb fresh start karte hain. Kya poochna hai?", parse_mode="HTML")

@bot.message_handler(commands=["stats"])
def cmd_stats(msg: types.Message):
    uid = msg.from_user.id
    name = msg.from_user.first_name or "User"
    tokens = user_tokens_used.get(uid, 0)
    msgs_count = len(user_histories.get(uid, [])) // 2
    first = user_first_seen.get(uid, "N/A")
    last = user_last_active.get(uid, "N/A")
    has_summary = "✅ Available" if uid in user_summaries else "❌ None"
    lang = user_languages.get(uid, "Auto detect")
    text = (
        f"📊 <b>{name} ki Stats</b>\n\n"
        f"🔑 <b>Tokens used:</b> {tokens:,}\n"
        f"💬 <b>Messages in memory:</b> {msgs_count}\n"
        f"📝 <b>Long-term summary:</b> {has_summary}\n"
        f"🌐 <b>Language:</b> {lang}\n"
        f"📅 <b>First seen:</b> {first}\n"
        f"🕐 <b>Last active:</b> {last}\n"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")

@bot.message_handler(commands=["summary"])
def cmd_summary(msg: types.Message):
    uid = msg.from_user.id
    send_typing(msg.chat.id)
    if uid in user_summaries:
        bot.send_message(msg.chat.id, f"📝 <b>Conversation Summary:</b>\n\n{clean_for_telegram(user_summaries[uid])}", parse_mode="HTML")
    elif user_histories.get(uid):
        m = bot.send_message(msg.chat.id, "⏳ Summary ban rahi hai...", parse_mode="HTML")
        summarise_history(uid)
        try:
            bot.delete_message(msg.chat.id, m.message_id)
        except Exception:
            pass
        if uid in user_summaries:
            bot.send_message(msg.chat.id, f"📝 <b>Summary:</b>\n\n{clean_for_telegram(user_summaries[uid])}", parse_mode="HTML")
        else:
            bot.send_message(msg.chat.id, "⚠️ Summary generate nahi ho saki.", parse_mode="HTML")
    else:
        bot.send_message(msg.chat.id, "ℹ️ Abhi koi conversation history nahi hai.\n\nPehle kuch baat karo!", parse_mode="HTML")

@bot.message_handler(commands=["translate"])
def cmd_translate(msg: types.Message):
    uid = msg.from_user.id
    parts = msg.text.split(None, 2)
    if len(parts) < 3:
        kb = types.InlineKeyboardMarkup(row_width=3)
        btns = [types.InlineKeyboardButton(lang, callback_data=f"translang_{code}")
                for lang, code in list(SUPPORTED_LANGS.items())[:24]]
        kb.add(*btns)
        bot.send_message(msg.chat.id,
            "🌐 <b>Translation</b>\n\n"
            "<b>Usage:</b> /translate [language] [text]\n\n"
            "<b>Example:</b>\n"
            "/translate Hindi Hello, how are you?\n"
            "/translate French Mujhe khana chahiye\n\n"
            "Ya neeche se language choose karo:",
            parse_mode="HTML", reply_markup=kb)
        return
    lang = parts[1]
    text_to_translate = parts[2]
    update_user_meta(uid, msg.from_user.first_name or "")
    send_typing(msg.chat.id)
    loading = bot.send_message(msg.chat.id, f"🌐 {lang} mein translate kar raha hun...")
    result = ask_ai(uid, f"Translate the following text to {lang}. Only output the translation, nothing else:\n\n{text_to_translate}")
    try:
        bot.delete_message(msg.chat.id, loading.message_id)
    except Exception:
        pass
    bot.send_message(msg.chat.id, f"🌐 <b>Translation ({lang}):</b>\n\n{clean_for_telegram(result)}", parse_mode="HTML")

@bot.message_handler(commands=["language"])
def cmd_language(msg: types.Message):
    kb = types.InlineKeyboardMarkup(row_width=3)
    btns = [types.InlineKeyboardButton(lang, callback_data=f"setlang_{code}")
            for lang, code in list(SUPPORTED_LANGS.items())[:30]]
    kb.add(*btns)
    bot.send_message(msg.chat.id, "🌐 <b>Apni preferred language choose karo:</b>\n\nBot is language mein jawab dega.", parse_mode="HTML", reply_markup=kb)

@bot.message_handler(commands=["leaderboard"])
def cmd_leaderboard(msg: types.Message):
    sorted_users = sorted(user_tokens_used.items(), key=lambda x: x[1], reverse=True)[:10]
    if not sorted_users:
        bot.send_message(msg.chat.id, "📊 Abhi koi data nahi hai.", parse_mode="HTML")
        return
    medals = ["🥇", "🥈", "🥉"] + ["🔹"] * 7
    lines = ["🏆 <b>User Leaderboard (by tokens used)</b>\n"]
    for i, (uid, tokens) in enumerate(sorted_users):
        name = user_names.get(uid, f"User {uid}")
        lines.append(f"{medals[i]} <b>{name}</b> — {tokens:,} tokens")
    bot.send_message(msg.chat.id, "\n".join(lines), parse_mode="HTML")

# ══════════════════════════════════════════════════════
#  ADMIN COMMANDS
# ══════════════════════════════════════════════════════
def is_admin(uid: int) -> bool:
    return uid in ADMIN_IDS

@bot.message_handler(commands=["admin"])
def cmd_admin(msg: types.Message):
    if not is_admin(msg.from_user.id):
        bot.send_message(msg.chat.id, "🚫 Tujhe admin access nahi hai.")
        return
    total = len(user_first_seen)
    today_str = today()
    active = sum(1 for dt in user_last_active.values() if dt.startswith(today_str))
    daily = daily_messages.get(today_str, 0)
    total_tok = sum(user_tokens_used.values())
    avg_rt = round(sum(response_times) / len(response_times), 2) if response_times else 0
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("👥 Total Users", callback_data="adm_users"),
        types.InlineKeyboardButton("🟢 Active Today", callback_data="adm_active"),
        types.InlineKeyboardButton("📨 Daily Messages", callback_data="adm_daily"),
        types.InlineKeyboardButton("⚠️ Error Logs", callback_data="adm_errors"),
        types.InlineKeyboardButton("⏱ Avg Response", callback_data="adm_resp"),
        types.InlineKeyboardButton("🤖 Model Stats", callback_data="adm_model"),
    )
    text = (
        f"👑 <b>Admin Panel</b>\n\n"
        f"👥 Total Users: <b>{total}</b>\n"
        f"🟢 Active Today: <b>{active}</b>\n"
        f"📨 Daily Messages: <b>{daily}</b>\n"
        f"🔑 Total Tokens: <b>{total_tok:,}</b>\n"
        f"⏱ Avg Response: <b>{avg_rt}s</b>\n"
        f"⚠️ Errors Logged: <b>{len(error_logs)}</b>\n"
        f"🤖 Model: <b>{MODEL}</b>"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML", reply_markup=kb)

@bot.message_handler(commands=["totalusers"])
def cmd_totalusers(msg: types.Message):
    if not is_admin(msg.from_user.id): return
    bot.send_message(msg.chat.id, f"👥 <b>Total Users:</b> {len(user_first_seen)}", parse_mode="HTML")

@bot.message_handler(commands=["activeusers"])
def cmd_activeusers(msg: types.Message):
    if not is_admin(msg.from_user.id): return
    active = sum(1 for dt in user_last_active.values() if dt.startswith(today()))
    bot.send_message(msg.chat.id, f"🟢 <b>Active Users Today:</b> {active}", parse_mode="HTML")

@bot.message_handler(commands=["dailymsgs"])
def cmd_dailymsgs(msg: types.Message):
    if not is_admin(msg.from_user.id): return
    bot.send_message(msg.chat.id, f"📨 <b>Today's Messages:</b> {daily_messages.get(today(), 0)}", parse_mode="HTML")

@bot.message_handler(commands=["errors"])
def cmd_errors(msg: types.Message):
    if not is_admin(msg.from_user.id): return
    if not error_logs:
        bot.send_message(msg.chat.id, "✅ <b>No errors logged!</b>", parse_mode="HTML")
        return
    lines = [f"• [{e['time'][11:19]}] {e['error'][:150]}" for e in error_logs[-10:]]
    bot.send_message(msg.chat.id, "⚠️ <b>Recent Errors:</b>\n\n" + "\n\n".join(lines), parse_mode="HTML")

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
        f"Errors: <b>{len(error_logs)}</b>\n"
        f"Response samples: <b>{len(response_times)}</b>"
    )
    bot.send_message(msg.chat.id, text, parse_mode="HTML")

# ══════════════════════════════════════════════════════
#  CALLBACK QUERY HANDLER
# ══════════════════════════════════════════════════════
@bot.callback_query_handler(func=lambda c: True)
def handle_callback(call: types.CallbackQuery):
    uid = call.from_user.id
    data = call.data

    if data.startswith("setlang_"):
        code = data.replace("setlang_", "")
        user_languages[uid] = code
        lang_name = next((k for k, v in SUPPORTED_LANGS.items() if v == code), code)
        bot.answer_callback_query(call.id, f"✅ Language set: {lang_name}")
        try:
            bot.edit_message_text(
                f"✅ <b>Language set to {lang_name}!</b>\n\nAb main {lang_name} mein jawab dunga.",
                call.message.chat.id, call.message.message_id, parse_mode="HTML"
            )
        except Exception:
            pass

    elif data.startswith("translang_"):
        code = data.replace("translang_", "")
        lang_name = next((k for k, v in SUPPORTED_LANGS.items() if v == code), code)
        bot.answer_callback_query(call.id, f"Selected: {lang_name}")
        try:
            bot.edit_message_text(
                f"🌐 <b>{lang_name} translation:</b>\n\nUse: /translate {lang_name} [your text]",
                call.message.chat.id, call.message.message_id, parse_mode="HTML"
            )
        except Exception:
            pass

    elif data.startswith("continue_"):
        target_uid = int(data.replace("continue_", ""))
        if uid == target_uid:
            send_typing(call.message.chat.id)
            bot.answer_callback_query(call.id, "▶️ Continuing...")
            result = ask_ai(uid, "Please continue your previous response from where you left off. Don't repeat what you already said.")
            cleaned = clean_for_telegram(result)
            parts = split_long_message(cleaned)
            for part in parts:
                bot.send_message(call.message.chat.id, part, parse_mode="HTML")
        else:
            bot.answer_callback_query(call.id, "❌ Ye tumhara button nahi hai.")

    elif data in ["adm_users", "adm_active", "adm_daily", "adm_errors", "adm_resp", "adm_model"]:
        if not is_admin(uid):
            bot.answer_callback_query(call.id, "🚫 Access denied")
            return
        if data == "adm_users":
            bot.answer_callback_query(call.id, f"Total Users: {len(user_first_seen)}", show_alert=True)
        elif data == "adm_active":
            active = sum(1 for dt in user_last_active.values() if dt.startswith(today()))
            bot.answer_callback_query(call.id, f"Active Today: {active}", show_alert=True)
        elif data == "adm_daily":
            bot.answer_callback_query(call.id, f"Today's Messages: {daily_messages.get(today(), 0)}", show_alert=True)
        elif data == "adm_errors":
            bot.answer_callback_query(call.id, f"Errors: {len(error_logs)}", show_alert=True)
        elif data == "adm_resp":
            avg = round(sum(response_times)/len(response_times), 2) if response_times else 0
            bot.answer_callback_query(call.id, f"Avg Response: {avg}s", show_alert=True)
        elif data == "adm_model":
            total = sum(user_tokens_used.values())
            bot.answer_callback_query(call.id, f"Total Tokens: {total:,}", show_alert=True)

# ══════════════════════════════════════════════════════
#  DOCUMENT HANDLER
# ══════════════════════════════════════════════════════
@bot.message_handler(content_types=["document"])
def handle_document(msg: types.Message):
    uid = msg.from_user.id
    update_user_meta(uid, msg.from_user.first_name or "")
    increment_daily()

    doc = msg.document
    fname = doc.file_name or "file"
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""

    loading = bot.send_message(msg.chat.id, f"📂 <b>{fname}</b> process ho rahi hai...", parse_mode="HTML")
    send_typing(msg.chat.id)

    try:
        file_info = bot.get_file(doc.file_id)
        file_bytes = bot.download_file(file_info.file_path)

        if ext == "pdf":
            content = process_pdf(file_bytes); file_type = "PDF"
        elif ext == "docx":
            content = process_docx(file_bytes); file_type = "Word Document"
        elif ext == "txt":
            content = process_txt(file_bytes); file_type = "Text File"
        elif ext == "csv":
            content = process_csv(file_bytes); file_type = "CSV File"
        elif ext in ("xlsx", "xls"):
            content = process_excel(file_bytes); file_type = "Excel File"
        elif ext == "json":
            content = process_json(file_bytes); file_type = "JSON File"
        elif ext == "zip":
            content = process_zip(file_bytes); file_type = "ZIP Archive"
        elif ext in ("py","js","ts","java","cpp","c","cs","go","rb","php","html","css","sql","sh","yaml","yml","xml","md","kt","swift","rs","dart"):
            content = process_code(file_bytes); file_type = f"Code File (.{ext})"
        else:
            content = process_txt(file_bytes); file_type = "File"

        caption = msg.caption or ""
        user_question = caption if caption else f"Is {file_type} ko analyse karo aur key points, summary, aur important information batao."
        extra_ctx = f"[Uploaded File: {fname} | Type: {file_type}]\n\nFile Content:\n{content}"

        try:
            bot.delete_message(msg.chat.id, loading.message_id)
        except Exception:
            pass

        result = ask_ai(uid, user_question, extra_ctx)
        reaction = get_reaction(result)

        # Add continue button if response might be truncated
        kb = None
        if len(result) > 1500:
            kb = types.InlineKeyboardMarkup()
            kb.add(types.InlineKeyboardButton("▶️ Continue response", callback_data=f"continue_{uid}"))

        cleaned = clean_for_telegram(result)
        parts = split_long_message(cleaned)
        try:
            bot.set_message_reaction(msg.chat.id, msg.message_id, [types.ReactionTypeEmoji(reaction)])
        except Exception:
            pass
        for i, part in enumerate(parts):
            markup = kb if (i == len(parts) - 1 and kb) else None
            try:
                bot.send_message(msg.chat.id, part, parse_mode="HTML", reply_markup=markup)
            except Exception:
                bot.send_message(msg.chat.id, re.sub(r"<[^>]+>", "", part))

    except Exception as e:
        record_error(str(e))
        log.error("Doc error: %s", e)
        try:
            bot.delete_message(msg.chat.id, loading.message_id)
        except Exception:
            pass
        bot.send_message(msg.chat.id, f"❌ File process nahi ho saki.\n\nError: {str(e)[:150]}", parse_mode="HTML")

# ══════════════════════════════════════════════════════
#  VOICE HANDLER
# ══════════════════════════════════════════════════════
@bot.message_handler(content_types=["voice"])
def handle_voice(msg: types.Message):
    uid = msg.from_user.id
    update_user_meta(uid, msg.from_user.first_name or "")
    increment_daily()
    loading = bot.send_message(msg.chat.id, "🎙️ Voice message process ho raha hai...", parse_mode="HTML")
    try:
        file_info = bot.get_file(msg.voice.file_id)
        audio_bytes = bot.download_file(file_info.file_path)
        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            with open(tmp_path, "rb") as af:
                transcript = client.audio.transcriptions.create(model="openai/whisper-large-v3", file=af)
            transcribed = transcript.text
            os.unlink(tmp_path)
            try:
                bot.delete_message(msg.chat.id, loading.message_id)
            except Exception:
                pass
            bot.send_message(msg.chat.id, f"🎙️ <b>Suna:</b> {transcribed}", parse_mode="HTML")
            result = ask_ai(uid, transcribed)
            send_reply(msg, result, get_reaction(result))
        except Exception:
            try:
                bot.delete_message(msg.chat.id, loading.message_id)
            except Exception:
                pass
            bot.send_message(msg.chat.id,
                "🎙️ Voice message mila!\n\n"
                "⚠️ Abhi automatic transcription available nahi hai.\n"
                "Kripya apna message <b>text mein</b> type karke bhejo.",
                parse_mode="HTML")
    except Exception as e:
        record_error(str(e))
        try:
            bot.delete_message(msg.chat.id, loading.message_id)
        except Exception:
            pass
        bot.send_message(msg.chat.id, "❌ Voice process nahi ho saka.", parse_mode="HTML")

# ══════════════════════════════════════════════════════
#  PHOTO HANDLER
# ══════════════════════════════════════════════════════
@bot.message_handler(content_types=["photo"])
def handle_photo(msg: types.Message):
    uid = msg.from_user.id
    update_user_meta(uid, msg.from_user.first_name or "")
    increment_daily()
    caption = msg.caption or "Is image mein kya hai? Detail mein describe karo."
    loading = bot.send_message(msg.chat.id, "🖼️ Image dekh raha hun...", parse_mode="HTML")
    send_typing(msg.chat.id)
    try:
        photo = msg.photo[-1]
        file_info = bot.get_file(photo.file_id)
        img_bytes = bot.download_file(file_info.file_path)
        b64 = base64.b64encode(img_bytes).decode()
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": caption}
                ]}
            ],
            max_tokens=MAX_TOKENS,
            extra_headers={"HTTP-Referer": "https://t.me/", "X-Title": "Telegram AI Bot"},
        )
        answer = resp.choices[0].message.content or "Image samajh nahi aaya."
        try:
            bot.delete_message(msg.chat.id, loading.message_id)
        except Exception:
            pass
        send_reply(msg, answer, "🖼️")
    except Exception as e:
        record_error(str(e))
        try:
            bot.delete_message(msg.chat.id, loading.message_id)
        except Exception:
            pass
        result = ask_ai(uid, caption + "\n[Note: User sent an image but vision processing failed. Respond helpfully.]")
        send_reply(msg, result, "🖼️")

# ══════════════════════════════════════════════════════
#  GROUP ADMIN TOOLS
# ══════════════════════════════════════════════════════
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
            bot.send_message(cid, "🚫 Sirf group admins ye command use kar sakte hain.")
            return
        if not msg.reply_to_message:
            bot.send_message(cid, "ℹ️ Kisi user ke message ko reply karke ye command use karo.")
            return
        target = msg.reply_to_message.from_user
        cmd = msg.text.split()[0][1:].lower()
        if cmd == "kick":
            bot.ban_chat_member(cid, target.id)
            bot.unban_chat_member(cid, target.id)
            bot.send_message(cid, f"👢 <b>{target.first_name}</b> ko group se kick kar diya gaya.", parse_mode="HTML")
        elif cmd == "ban":
            bot.ban_chat_member(cid, target.id)
            bot.send_message(cid, f"🔨 <b>{target.first_name}</b> ko permanently ban kar diya gaya.", parse_mode="HTML")
        elif cmd == "mute":
            until = int(time.time()) + 86400
            bot.restrict_chat_member(cid, target.id, types.ChatPermissions(can_send_messages=False), until_date=until)
            bot.send_message(cid, f"🔇 <b>{target.first_name}</b> ko 24 ghante ke liye mute kar diya.", parse_mode="HTML")
        elif cmd == "unmute":
            bot.restrict_chat_member(cid, target.id, types.ChatPermissions(
                can_send_messages=True, can_send_media_messages=True,
                can_send_polls=True, can_send_other_messages=True))
            bot.send_message(cid, f"🔊 <b>{target.first_name}</b> unmute ho gaya.", parse_mode="HTML")
        elif cmd == "warn":
            bot.send_message(cid, f"⚠️ <b>{target.first_name}</b> ko warning mili!\n\nBehaviour theek karo warna ban hoga. 🚫", parse_mode="HTML")
    except Exception as e:
        record_error(str(e))
        bot.send_message(cid, f"❌ Action fail hua: {str(e)[:100]}", parse_mode="HTML")

# ══════════════════════════════════════════════════════
#  MAIN TEXT HANDLER
# ══════════════════════════════════════════════════════
@bot.message_handler(func=lambda m: m.text and not m.text.startswith("/"))
def handle_text(msg: types.Message):
    uid = msg.from_user.id
    name = msg.from_user.first_name or "User"
    text = msg.text.strip()

    update_user_meta(uid, name)
    increment_daily()

    if not check_rate_limit(uid):
        bot.send_message(msg.chat.id,
            "⏳ <b>Thoda ruk ja bhai!</b>\n\n"
            "Bahut fast messages aa rahe hain. 1 minute mein maximum 100 messages allowed hain.\n"
            "Thodi der baad phir try karo! 😊",
            parse_mode="HTML")
        return

    send_typing(msg.chat.id)

    # Add language preference if set
    lang_pref = user_languages.get(uid, "")
    extra = f"[User's preferred reply language: {lang_pref}. Always reply in this language.]\n" if lang_pref else ""

    loading = bot.send_message(msg.chat.id, "🧠 Soch raha hun...", parse_mode="HTML")
    result = ask_ai(uid, text, extra)

    try:
        bot.delete_message(msg.chat.id, loading.message_id)
    except Exception:
        pass

    reaction = get_reaction(text + " " + result)

    # Continue button if long response
    kb = None
    if len(result) > 1800:
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton("▶️ Continue response", callback_data=f"continue_{uid}"))

    cleaned = clean_for_telegram(result)
    parts = split_long_message(cleaned)

    try:
        bot.set_message_reaction(msg.chat.id, msg.message_id, [types.ReactionTypeEmoji(reaction)])
    except Exception:
        pass

    for i, part in enumerate(parts):
        markup = kb if (i == len(parts) - 1 and kb) else None
        try:
            bot.send_message(msg.chat.id, part, parse_mode="HTML", reply_markup=markup)
        except Exception as e:
            record_error(str(e))
            try:
                bot.send_message(msg.chat.id, re.sub(r"<[^>]+>", "", part))
            except Exception:
                pass

# ══════════════════════════════════════════════════════
#  FLASK WEBHOOK
# ══════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def index():
    return "🤖 LyraBot is running!", 200

@app.route("/health", methods=["GET"])
def health():
    return json.dumps({
        "status": "ok",
        "users": len(user_first_seen),
        "model": MODEL,
        "errors": len(error_logs),
    }), 200

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        json_str = request.get_data(as_text=True)
        update = types.Update.de_json(json_str)
        bot.process_new_updates([update])
    return "OK", 200

# ══════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════
def setup_webhook():
    if WEBHOOK_URL:
        url = f"{WEBHOOK_URL.rstrip('/')}/{BOT_TOKEN}"
        bot.remove_webhook()
        time.sleep(1)
        bot.set_webhook(url=url)
        log.info("✅ Webhook set: %s", url)
    else:
        log.warning("⚠️ WEBHOOK_URL not set — polling mode")

if __name__ == "__main__":
    if not BOT_TOKEN:
        log.error("❌ BOT_TOKEN not set!")
        exit(1)
    if not OPENROUTER_KEY:
        log.error("❌ OPENROUTER_KEY not set!")
        exit(1)
    if WEBHOOK_URL:
        setup_webhook()
        port = int(os.environ.get("PORT", 10000))
        log.info("🚀 Starting Flask on port %s", port)
        app.run(host="0.0.0.0", port=port)
    else:
        log.info("🔄 Starting polling mode...")
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
