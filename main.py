import os
import io
import json
import time
import zipfile
import logging
import tempfile
import datetime
import threading
import traceback
from collections import defaultdict

import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, ReactionTypeEmoji
from flask import Flask, request
from openai import OpenAI

# ─── ENV VARS ───────────────────────────────────────────────────────────────
BOT_TOKEN  = os.environ["BOT_TOKEN"]
API_KEY    = os.environ["API_KEY"]          # your sk-nara-... key
BASE_URL   = os.environ.get("BASE_URL", "https://router.bynara.id/v1")
MODEL      = os.environ.get("MODEL", "auto/bynara")
ADMIN_IDS  = [int(x) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip()]

# ─── INIT ────────────────────────────────────────────────────────────────────
bot    = telebot.TeleBot(BOT_TOKEN, parse_mode="Markdown")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
app    = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── IN-MEMORY STORES ────────────────────────────────────────────────────────
# Long-term memory: user_id → list of {role, content} (full history, trimmed)
memory: dict[int, list[dict]] = defaultdict(list)
# Summaries for old context
summaries: dict[int, str] = {}
# Stats
stats = {
    "total_users":    set(),
    "active_today":   set(),
    "daily_messages": defaultdict(int),
    "token_usage":    defaultdict(int),
    "response_times": [],
    "errors":         [],
    "model_usage":    defaultdict(int),
    "user_messages":  defaultdict(int),   # leaderboard
}
# Pending "continue" responses (user_id → leftover text)
pending_continue: dict[int, str] = {}

# ─── HELPERS ─────────────────────────────────────────────────────────────────
SUPPORTED_LANGS = {
    "en":"English","ur":"Urdu","hi":"Hindi","ar":"Arabic","fr":"French",
    "de":"German","es":"Spanish","it":"Italian","pt":"Portuguese","ru":"Russian",
    "zh":"Chinese","ja":"Japanese","ko":"Korean","tr":"Turkish","nl":"Dutch",
    "pl":"Polish","sv":"Swedish","da":"Danish","fi":"Finnish","no":"Norwegian",
    "bn":"Bengali","pa":"Punjabi","fa":"Persian","id":"Indonesian","ms":"Malay",
    "th":"Thai","vi":"Vietnamese","ro":"Romanian","hu":"Hungarian","cs":"Czech",
}

MAX_HISTORY   = 20          # messages kept per user before summarising
MAX_TG_LENGTH = 4000        # Telegram message char limit (safe)


def today_str():
    return datetime.date.today().isoformat()


def record_stat(user_id: int, tokens: int = 0, resp_time: float = 0):
    stats["total_users"].add(user_id)
    stats["active_today"].add(user_id)
    stats["daily_messages"][today_str()] += 1
    stats["token_usage"][today_str()] += tokens
    stats["user_messages"][user_id] += 1
    if resp_time:
        stats["response_times"].append(resp_time)
    stats["model_usage"][MODEL] += 1


def log_error(user_id: int, err: str):
    stats["errors"].append({
        "time": datetime.datetime.utcnow().isoformat(),
        "user": user_id,
        "error": err,
    })
    logger.error("[Error uid=%s] %s", user_id, err)


def get_system_prompt(lang_hint: str = "auto") -> str:
    return (
        f"You are an advanced, helpful AI assistant inside Telegram. "
        f"Detected/preferred language: {lang_hint}. "
        "Reply in the same language the user uses. "
        "Be concise but thorough. Use markdown formatting when helpful."
    )


def detect_language(text: str) -> str:
    """Very lightweight heuristic; full detection done by the model itself."""
    if any("\u0600" <= c <= "\u06ff" for c in text):
        return "Arabic/Urdu/Persian"
    if any("\u0900" <= c <= "\u097f" for c in text):
        return "Hindi"
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return "Chinese"
    if any("\u3040" <= c <= "\u30ff" for c in text):
        return "Japanese"
    if any("\uac00" <= c <= "\ud7a3" for c in text):
        return "Korean"
    return "Latin-script"


def maybe_summarise(user_id: int):
    """If history is long, summarise old turns and reset."""
    hist = memory[user_id]
    if len(hist) < MAX_HISTORY:
        return
    to_summarise = hist[:-4]   # keep last 4 turns fresh
    memory[user_id] = hist[-4:]
    joined = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in to_summarise)
    try:
        r = client.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "Summarise the following conversation in 3-5 sentences, keeping key facts."},
                {"role": "user",   "content": joined},
            ],
        )
        summaries[user_id] = r.choices[0].message.content.strip()
    except Exception as e:
        log_error(user_id, f"summarise: {e}")


def build_messages(user_id: int, user_text: str, lang: str = "auto") -> list[dict]:
    maybe_summarise(user_id)
    system = get_system_prompt(lang)
    if user_id in summaries:
        system += f"\n\n[Earlier conversation summary]: {summaries[user_id]}"
    messages = [{"role": "system", "content": system}]
    messages += memory[user_id]
    messages.append({"role": "user", "content": user_text})
    return messages


def call_ai(user_id: int, user_text: str, lang: str = "auto") -> tuple[str, int]:
    """Returns (reply_text, tokens_used)."""
    messages = build_messages(user_id, user_text, lang)
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=MODEL,
            max_tokens=2000,
            messages=messages,
        )
        reply  = r.choices[0].message.content.strip()
        tokens = r.usage.total_tokens if r.usage else 0
        record_stat(user_id, tokens, time.time() - t0)
        # Store in memory
        memory[user_id].append({"role": "user",      "content": user_text})
        memory[user_id].append({"role": "assistant",  "content": reply})
        return reply, tokens
    except Exception as e:
        log_error(user_id, traceback.format_exc())
        raise e


def send_long(chat_id: int, text: str, reply_to: int | None = None, user_id: int | None = None):
    """Send text, chunking if needed, with a 'Continue' button for very long replies."""
    if len(text) <= MAX_TG_LENGTH:
        bot.send_message(chat_id, text, reply_to_message_id=reply_to)
        return
    # Split at MAX_TG_LENGTH
    chunk   = text[:MAX_TG_LENGTH]
    leftover = text[MAX_TG_LENGTH:]
    if user_id:
        pending_continue[user_id] = leftover
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton("▶️ Continue", callback_data="continue_response"))
    bot.send_message(chat_id, chunk, reply_to_message_id=reply_to, reply_markup=kb)


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from various file types."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")

    if ext == "json":
        try:
            obj = json.loads(file_bytes)
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")

    if ext == "csv":
        import csv
        text = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        preview = f"CSV File: {filename}\nRows: {len(rows)}, Cols: {len(rows[0]) if rows else 0}\n\n"
        preview += "\n".join([", ".join(r) for r in rows[:20]])
        if len(rows) > 20:
            preview += f"\n... ({len(rows)-20} more rows)"
        return preview

    if ext in ("xls", "xlsx"):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True)
            out = []
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                out.append(f"## Sheet: {sheet}")
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    if i > 50: out.append("... (truncated)"); break
                    out.append("\t".join([str(c) if c is not None else "" for c in row]))
            return "\n".join(out)
        except Exception as e:
            return f"[Could not parse Excel: {e}]"

    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = []
            for i, page in enumerate(reader.pages[:15]):
                pages.append(f"[Page {i+1}]\n{page.extract_text()}")
                if i >= 14:
                    pages.append("... (truncated to 15 pages)")
                    break
            return "\n\n".join(pages)
        except Exception as e:
            return f"[Could not parse PDF: {e}]"

    if ext in ("doc", "docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception as e:
            return f"[Could not parse DOCX: {e}]"

    if ext == "zip":
        try:
            zf = zipfile.ZipFile(io.BytesIO(file_bytes))
            names = zf.namelist()
            out = [f"ZIP Contents ({len(names)} files):"]
            for n in names[:30]:
                out.append(f"  • {n}")
            if len(names) > 30:
                out.append(f"  ... ({len(names)-30} more)")
            # Try to read text files inside
            text_files = [n for n in names if n.endswith((".txt",".py",".js",".json",".md",".csv"))][:5]
            for tf in text_files:
                content = zf.read(tf).decode("utf-8", errors="ignore")[:500]
                out.append(f"\n--- {tf} ---\n{content}")
            return "\n".join(out)
        except Exception as e:
            return f"[Could not read ZIP: {e}]"

    # Code files
    code_exts = {"py","js","ts","java","c","cpp","cs","go","rb","php","html","css","sh","yaml","yml","toml","ini","xml","md","rs","kt","swift"}
    if ext in code_exts:
        return file_bytes.decode("utf-8", errors="ignore")[:4000]

    return f"[Unsupported file type: .{ext}]"


# ─── REACTIONS ───────────────────────────────────────────────────────────────
REACTIONS = ["👍", "🔥", "🤩", "💯", "🎉"]

def add_reaction(chat_id: int, message_id: int, emoji: str = "👍"):
    try:
        bot.set_message_reaction(chat_id, message_id, [ReactionTypeEmoji(emoji)])
    except Exception:
        pass  # Reactions may not be supported in all chats


# ─── COMMAND HANDLERS ────────────────────────────────────────────────────────
@bot.message_handler(commands=["start"])
def cmd_start(msg):
    stats["total_users"].add(msg.from_user.id)
    add_reaction(msg.chat.id, msg.message_id, "👋")
    bot.reply_to(msg,
        "🤖 *Welcome to AI Assistant Bot!*\n\n"
        "I can help you with:\n"
        "📄 File analysis (PDF, DOCX, TXT, CSV, Excel, JSON, ZIP, Code)\n"
        "🎤 Voice messages\n"
        "🌍 100+ language translation\n"
        "🧠 Long-term memory\n"
        "💬 Smart conversations\n\n"
        "Just send me a message, file, or voice note!\n\n"
        "Commands: /help /stats /clear /translate /admin"
    )


@bot.message_handler(commands=["help"])
def cmd_help(msg):
    add_reaction(msg.chat.id, msg.message_id, "💡")
    bot.reply_to(msg,
        "*📚 Commands:*\n"
        "/start — Welcome\n"
        "/help — This message\n"
        "/clear — Clear your conversation memory\n"
        "/translate `<lang_code> <text>` — Translate text\n"
        "/stats — Your personal stats\n"
        "/admin — Admin panel (admins only)\n\n"
        "*💡 Tips:*\n"
        "• Send any file for AI analysis\n"
        "• Send voice messages — I'll transcribe & reply\n"
        "• I remember our full conversation history\n"
        "• I auto-detect your language"
    )


@bot.message_handler(commands=["clear"])
def cmd_clear(msg):
    uid = msg.from_user.id
    memory[uid].clear()
    summaries.pop(uid, None)
    add_reaction(msg.chat.id, msg.message_id, "🗑")
    bot.reply_to(msg, "✅ Your conversation memory has been cleared!")


@bot.message_handler(commands=["translate"])
def cmd_translate(msg):
    uid = msg.from_user.id
    parts = msg.text.split(maxsplit=2)
    if len(parts) < 3:
        bot.reply_to(msg, "Usage: `/translate <lang_code> <text>`\n\nExamples:\n`/translate fr Hello world`\n`/translate ur Good morning`\n\nAvailable: " + ", ".join(f"`{k}`={v}" for k,v in list(SUPPORTED_LANGS.items())[:10]) + " ...")
        return
    lang_code = parts[1].lower()
    text      = parts[2]
    lang_name = SUPPORTED_LANGS.get(lang_code, lang_code)
    try:
        reply, _ = call_ai(uid, f"Translate the following text to {lang_name}. Output ONLY the translation:\n\n{text}", lang=lang_name)
        add_reaction(msg.chat.id, msg.message_id, "🌍")
        bot.reply_to(msg, f"*Translation ({lang_name}):*\n{reply}")
    except Exception as e:
        bot.reply_to(msg, f"❌ Translation failed: {e}")


@bot.message_handler(commands=["stats"])
def cmd_stats(msg):
    uid  = msg.from_user.id
    msgs = stats["user_messages"].get(uid, 0)
    add_reaction(msg.chat.id, msg.message_id, "📊")
    bot.reply_to(msg,
        f"*📊 Your Stats:*\n"
        f"Messages sent: {msgs}\n"
        f"Memory turns: {len(memory.get(uid, []))}\n"
        f"Has summary: {'Yes' if uid in summaries else 'No'}"
    )


@bot.message_handler(commands=["admin"])
def cmd_admin(msg):
    uid = msg.from_user.id
    if uid not in ADMIN_IDS:
        bot.reply_to(msg, "❌ You are not an admin.")
        return
    add_reaction(msg.chat.id, msg.message_id, "🔐")
    today = today_str()
    avg_rt = (sum(stats["response_times"][-100:]) / max(len(stats["response_times"][-100:]), 1))
    # Leaderboard top 5
    lb = sorted(stats["user_messages"].items(), key=lambda x: x[1], reverse=True)[:5]
    lb_text = "\n".join([f"  {i+1}. uid={uid}: {cnt} msgs" for i,(uid,cnt) in enumerate(lb)])
    err_count = len(stats["errors"])
    last_errs = stats["errors"][-3:]
    err_text = "\n".join([f"  [{e['time']}] uid={e['user']}: {e['error'][:60]}" for e in last_errs]) or "None"
    model_text = "\n".join([f"  {m}: {c}" for m,c in stats["model_usage"].items()]) or "None"
    bot.reply_to(msg,
        f"*🔐 Admin Panel*\n\n"
        f"👥 Total users: {len(stats['total_users'])}\n"
        f"🟢 Active today: {len(stats['active_today'])}\n"
        f"💬 Messages today: {stats['daily_messages'].get(today, 0)}\n"
        f"🪙 Tokens today: {stats['token_usage'].get(today, 0)}\n"
        f"⚡ Avg response time: {avg_rt:.2f}s\n"
        f"❌ Total errors: {err_count}\n\n"
        f"*🏆 Leaderboard:*\n{lb_text}\n\n"
        f"*🤖 Model usage:*\n{model_text}\n\n"
        f"*🔴 Recent errors:*\n{err_text}"
    )


# ─── GROUP ADMIN TOOLS ───────────────────────────────────────────────────────
@bot.message_handler(commands=["ban"])
def cmd_ban(msg):
    if not is_group_admin(msg):
        return
    if not msg.reply_to_message:
        bot.reply_to(msg, "Reply to a user's message to ban them.")
        return
    target = msg.reply_to_message.from_user.id
    try:
        bot.ban_chat_member(msg.chat.id, target)
        add_reaction(msg.chat.id, msg.message_id, "🚫")
        bot.reply_to(msg, f"✅ User {target} has been banned.")
    except Exception as e:
        bot.reply_to(msg, f"❌ Failed: {e}")


@bot.message_handler(commands=["unban"])
def cmd_unban(msg):
    if not is_group_admin(msg):
        return
    if not msg.reply_to_message:
        bot.reply_to(msg, "Reply to a user's message to unban them.")
        return
    target = msg.reply_to_message.from_user.id
    try:
        bot.unban_chat_member(msg.chat.id, target)
        bot.reply_to(msg, f"✅ User {target} has been unbanned.")
    except Exception as e:
        bot.reply_to(msg, f"❌ Failed: {e}")


@bot.message_handler(commands=["mute"])
def cmd_mute(msg):
    if not is_group_admin(msg):
        return
    if not msg.reply_to_message:
        bot.reply_to(msg, "Reply to a user's message to mute them.")
        return
    target = msg.reply_to_message.from_user.id
    try:
        bot.restrict_chat_member(
            msg.chat.id, target,
            permissions=telebot.types.ChatPermissions(can_send_messages=False),
            until_date=int(time.time()) + 3600,   # 1 hour
        )
        add_reaction(msg.chat.id, msg.message_id, "🔇")
        bot.reply_to(msg, f"✅ User {target} muted for 1 hour.")
    except Exception as e:
        bot.reply_to(msg, f"❌ Failed: {e}")


@bot.message_handler(commands=["pin"])
def cmd_pin(msg):
    if not is_group_admin(msg):
        return
    if not msg.reply_to_message:
        bot.reply_to(msg, "Reply to a message to pin it.")
        return
    try:
        bot.pin_chat_message(msg.chat.id, msg.reply_to_message.message_id)
        add_reaction(msg.chat.id, msg.message_id, "📌")
    except Exception as e:
        bot.reply_to(msg, f"❌ Failed: {e}")


def is_group_admin(msg) -> bool:
    try:
        member = bot.get_chat_member(msg.chat.id, msg.from_user.id)
        if member.status in ("administrator", "creator"):
            return True
        bot.reply_to(msg, "❌ You need to be an admin to use this.")
        return False
    except Exception:
        return False


# ─── VOICE MESSAGES ──────────────────────────────────────────────────────────
@bot.message_handler(content_types=["voice"])
def handle_voice(msg):
    uid = msg.from_user.id
    bot.send_chat_action(msg.chat.id, "typing")
    try:
        file_info = bot.get_file(msg.voice.file_id)
        voice_bytes = bot.download_file(file_info.file_path)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(voice_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=("voice.ogg", f, "audio/ogg"),
            )
        text = transcript.text.strip()
        os.unlink(tmp_path)
        if not text:
            bot.reply_to(msg, "❌ Could not transcribe the voice message.")
            return
        bot.reply_to(msg, f"🎤 *Transcription:*\n_{text}_")
        lang = detect_language(text)
        reply, _ = call_ai(uid, text, lang)
        add_reaction(msg.chat.id, msg.message_id, "🎙")
        send_long(msg.chat.id, reply, reply_to=msg.message_id, user_id=uid)
    except Exception as e:
        log_error(uid, traceback.format_exc())
        bot.reply_to(msg, f"❌ Voice processing error: {e}")


# ─── DOCUMENT / FILE HANDLER ─────────────────────────────────────────────────
@bot.message_handler(content_types=["document"])
def handle_document(msg):
    uid      = msg.from_user.id
    doc      = msg.document
    filename = doc.file_name or "file"
    caption  = msg.caption or f"Analyse this file: {filename}"
    bot.send_chat_action(msg.chat.id, "upload_document")
    try:
        file_info  = bot.get_file(doc.file_id)
        file_bytes = bot.download_file(file_info.file_path)
        extracted  = extract_text_from_file(file_bytes, filename)
        prompt     = f"{caption}\n\n[File: {filename}]\n\n{extracted[:6000]}"
        lang       = detect_language(caption)
        add_reaction(msg.chat.id, msg.message_id, "📄")
        reply, _   = call_ai(uid, prompt, lang)
        send_long(msg.chat.id, reply, reply_to=msg.message_id, user_id=uid)
    except Exception as e:
        log_error(uid, traceback.format_exc())
        bot.reply_to(msg, f"❌ File processing error: {e}")


# ─── PHOTO HANDLER ───────────────────────────────────────────────────────────
@bot.message_handler(content_types=["photo"])
def handle_photo(msg):
    uid     = msg.from_user.id
    caption = msg.caption or "Describe this image."
    bot.send_chat_action(msg.chat.id, "typing")
    try:
        # Get highest-res photo
        photo     = sorted(msg.photo, key=lambda p: p.file_size)[-1]
        file_info = bot.get_file(photo.file_id)
        img_bytes = bot.download_file(file_info.file_path)
        import base64
        b64 = base64.b64encode(img_bytes).decode()
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": caption},
            ]},
        ]
        r = client.chat.completions.create(model=MODEL, max_tokens=1000, messages=messages)
        reply = r.choices[0].message.content.strip()
        add_reaction(msg.chat.id, msg.message_id, "🖼")
        send_long(msg.chat.id, reply, reply_to=msg.message_id, user_id=uid)
    except Exception as e:
        log_error(uid, traceback.format_exc())
        bot.reply_to(msg, f"❌ Image processing error: {e}")


# ─── TEXT MESSAGE HANDLER ─────────────────────────────────────────────────────
@bot.message_handler(func=lambda m: m.content_type == "text" and not m.text.startswith("/"))
def handle_text(msg):
    uid  = msg.from_user.id
    text = msg.text.strip()
    if not text:
        return
    bot.send_chat_action(msg.chat.id, "typing")
    lang = detect_language(text)
    try:
        reply, _ = call_ai(uid, text, lang)
        add_reaction(msg.chat.id, msg.message_id, "💬")
        send_long(msg.chat.id, reply, reply_to=msg.message_id, user_id=uid)
    except Exception as e:
        log_error(uid, traceback.format_exc())
        bot.reply_to(msg, f"❌ Error: {e}")


# ─── CALLBACK: CONTINUE BUTTON ───────────────────────────────────────────────
@bot.callback_query_handler(func=lambda c: c.data == "continue_response")
def cb_continue(call):
    uid  = call.from_user.id
    rest = pending_continue.pop(uid, None)
    if not rest:
        bot.answer_callback_query(call.id, "Nothing more to show.")
        return
    bot.answer_callback_query(call.id)
    send_long(call.message.chat.id, rest, user_id=uid)


# ─── FLASK WEBHOOK ───────────────────────────────────────────────────────────
@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    update = telebot.types.Update.de_json(request.get_json())
    bot.process_new_updates([update])
    return "ok", 200


@app.route("/", methods=["GET"])
def index():
    return "Bot is running ✅", 200


# ─── STARTUP ─────────────────────────────────────────────────────────────────
def set_webhook():
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "")
    if render_url:
        url = f"{render_url}/{BOT_TOKEN}"
        bot.remove_webhook()
        time.sleep(0.5)
        bot.set_webhook(url=url)
        logger.info("Webhook set to %s", url)
    else:
        logger.warning("RENDER_EXTERNAL_URL not set — running in polling mode")
        threading.Thread(target=bot.infinity_polling, daemon=True).start()


if __name__ == "__main__":
    set_webhook()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
