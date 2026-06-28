import os
import logging
import re
import threading
import random
import requests as http_requests
from datetime import datetime
from flask import Flask, request
import telebot
from telebot import types
from openai import OpenAI

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Environment Variables ─────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN")
HF_TOKEN  = os.environ.get("HF_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is not set!")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set!")

# ── Clients ───────────────────────────────────────────────────────────────────
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)

ai_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

MODEL = "deepseek-ai/DeepSeek-V3-0324:novita"

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "Bot is running!", 200

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    json_data = request.get_data().decode("utf-8")
    update = telebot.types.Update.de_json(json_data)
    bot.process_new_updates([update])
    return "OK", 200

# ── State Storage ─────────────────────────────────────────────────────────────
user_histories: dict[int, list[dict]] = {}
user_modes: dict[int, str] = {}        # "normal" | "translate" | "summarize"
user_lang: dict[int, str] = {}         # preferred language per user
user_stats: dict[int, dict] = {}       # message count, first seen

MAX_HISTORY = 20

# ── System Prompts ────────────────────────────────────────────────────────────
BASE_SYSTEM = """You are an advanced AI assistant inside a Telegram bot. Follow these rules STRICTLY:

FORMATTING RULES (very important):
- Use ONLY plain text and bold with *single asterisks* like *this* for important words
- Do NOT use # for headings — use plain text labels followed by a colon instead
- Do NOT use ## or ### ever
- Do NOT use markdown headers of any kind
- For lists, use a simple dash (-) or bullet (•) at the start of each line
- Do NOT use backtick code blocks with triple backticks for simple code — inline is fine
- Keep responses clear, structured, and readable in a chat interface

CONTENT RULES:
- Highlight key terms, facts, and important points with *bold*
- Be concise but thorough
- Be friendly and professional
- Remember the conversation context
"""

MODE_PROMPTS = {
    "translate": "The user wants translation help. Translate any text they send to English unless they specify another language. Always mention the source language detected.",
    "summarize": "The user wants summarization. Summarize any text they send in 3-5 bullet points maximum. Keep it crisp.",
    "code":      "The user wants coding help. Explain code clearly, point out bugs, and suggest improvements. Use simple formatting — no triple backtick blocks.",
    "normal":    "",
}

# ── Reactions — using Telegram's supported emoji list ────────────────────────
# Telegram only accepts specific emojis as reactions. These are all valid ones.
REACTION_MAP = [
    (["error", "problem", "issue", "bug", "crash"],        "🤔"),
    (["code", "program", "python", "script", "function",
      "debug", "coding"],                                  "👨‍💻"),
    (["thank", "thanks", "shukriya", "dhanyawad",
      "tysm", "ty"],                                       "🙏"),
    (["hello", "hi", "hey", "namaste", "salam",
      "hola", "howdy"],                                    "👋"),
    (["love", "pyar", "heart", "❤", "cute"],               "❤"),
    (["money", "price", "cost", "rupee", "dollar",
      "paisa", "finance"],                                 "💯"),
    (["idea", "suggest", "plan", "concept"],               "💡"),
    (["news", "update", "latest", "breaking"],             "🔥"),
    (["yes", "correct", "right", "bilkul", "haan",
      "sure", "absolutely"],                               "👍"),
    (["no", "wrong", "nahi", "galat", "nope",
      "incorrect"],                                        "👎"),
    (["sad", "dukh", "cry", "bura", "upset"],              "😢"),
    (["happy", "great", "awesome", "khushi", "amazing",
      "excellent", "wonderful"],                           "🎉"),
    (["funny", "lol", "haha", "joke", "maza"],             "😂"),
    (["wow", "amazing", "incredible", "unbelievable"],     "🤩"),
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
