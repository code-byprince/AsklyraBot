import os
import logging
import re
import threading
from flask import Flask, request
import telebot
from openai import OpenAI

# ── Logging ──────────────────────────────────────────────────────────────────
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
bot = telebot.TeleBot(BOT_TOKEN)

ai_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

MODEL = "deepseek-ai/DeepSeek-V3-0324:novita"

# ── Flask App (for Render keep-alive / webhook) ───────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "✅ Bot is running!", 200

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    json_data = request.get_data().decode("utf-8")
    update = telebot.types.Update.de_json(json_data)
    bot.process_new_updates([update])
    return "OK", 200

# ── Per-user conversation history ─────────────────────────────────────────────
user_histories: dict[int, list[dict]] = {}

SYSTEM_PROMPT = (
    "You are an advanced, professional AI assistant integrated into Telegram. "
    "Your responses should be clear, helpful, and well-structured. "
    "For important terms, key facts, or critical information, wrap them in **double asterisks** "
    "so they appear bold in Telegram (MarkdownV2). "
    "Keep answers concise yet thorough. Use bullet points when listing items. "
    "Be friendly, direct, and avoid unnecessary filler text."
)

MAX_HISTORY = 20   # messages kept per user (system excluded)

# ── Reaction emojis mapped to topic keywords ──────────────────────────────────
REACTION_MAP = [
    (["error", "problem", "issue", "fix", "bug"],      "🔧"),
    (["code", "program", "python", "script", "function"], "💻"),
    (["thank", "thanks", "shukriya", "dhanyawad"],     "🙏"),
    (["hello", "hi", "hey", "namaste", "salam"],       "👋"),
    (["love", "pyar", "heart"],                        "❤️"),
    (["money", "price", "cost", "rupee", "dollar"],    "💰"),
    (["idea", "suggest", "plan"],                      "💡"),
    (["news", "update", "latest"],                     "📰"),
    (["yes", "correct", "right", "bilkul", "haan"],    "✅"),
    (["no", "wrong", "nahi", "galat"],                 "❌"),
]

DEFAULT_REACTIONS = ["🤖", "💬", "🧠", "⚡", "🌟"]

def pick_reaction(text: str) -> str:
    lowered = text.lower()
    for keywords, emoji in REACTION_MAP:
        if any(kw in lowered for kw in keywords):
            return emoji
    import random
    return random.choice(DEFAULT_REACTIONS)


# ── Markdown safety for Telegram MarkdownV2 ───────────────────────────────────
ESCAPE_CHARS = r"_[]()~`>#+-=|{}.!"

def escape_md(text: str) -> str:
    """Escape special chars for MarkdownV2, but preserve **bold** markers."""
    # Split on **...** blocks, escape non-bold parts
    parts = re.split(r"(\*\*.*?\*\*)", text, flags=re.DOTALL)
    result = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            inner = part[2:-2]
            # Escape inside bold too (except the asterisks themselves)
            inner_escaped = re.sub(r"([_\[\]()~`>#+=|{}.!\-])", r"\\\1", inner)
            result.append(f"*{inner_escaped}*")
        else:
            escaped = re.sub(r"([_\[\]()~`>#+=|{}.!\-\*])", r"\\\1", part)
            result.append(escaped)
    return "".join(result)


def send_typing(chat_id: int):
    try:
        bot.send_chat_action(chat_id, "typing")
    except Exception:
        pass


# ── AI response helper ────────────────────────────────────────────────────────
def get_ai_response(user_id: int, user_message: str) -> str:
    if user_id not in user_histories:
        user_histories[user_id] = []

    history = user_histories[user_id]
    history.append({"role": "user", "content": user_message})

    # Trim history
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        user_histories[user_id] = history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        response = ai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"AI API error: {e}")
        return "⚠️ Sorry, I couldn't process your request right now. Please try again."


# ── Bot Handlers ──────────────────────────────────────────────────────────────
@bot.message_handler(commands=["start"])
def handle_start(message):
    name = message.from_user.first_name or "there"
    welcome = (
        f"👋 *Hello, {name}\\!*\n\n"
        "I'm your *AI Assistant* powered by **DeepSeek**\\.\n\n"
        "🧠 I can help you with:\n"
        "• *Answering questions* on any topic\n"
        "• *Writing & editing* content\n"
        "• *Coding* help & debugging\n"
        "• *Analysis* & research\n\n"
        "Just *type your message* and I'll respond\\!\n\n"
        "📌 Commands:\n"
        "/start \\- Restart the bot\n"
        "/clear \\- Clear conversation history\n"
        "/help \\- Show this help message"
    )
    bot.reply_to(message, welcome, parse_mode="MarkdownV2")


@bot.message_handler(commands=["help"])
def handle_help(message):
    help_text = (
        "🤖 *Bot Commands:*\n\n"
        "/start \\- Start the bot\n"
        "/clear \\- Clear your chat history \\(fresh start\\)\n"
        "/help \\- Show this message\n\n"
        "💡 *Tips:*\n"
        "• Ask me *anything* — I remember your conversation\\!\n"
        "• I highlight **key points** in bold for clarity\\.\n"
        "• Use /clear to reset context anytime\\."
    )
    bot.reply_to(message, help_text, parse_mode="MarkdownV2")


@bot.message_handler(commands=["clear"])
def handle_clear(message):
    user_id = message.from_user.id
    user_histories[user_id] = []
    bot.reply_to(
        message,
        "🗑️ *Conversation cleared\\!* Starting fresh\\.",
        parse_mode="MarkdownV2",
    )


@bot.message_handler(func=lambda m: m.text and not m.text.startswith("/"))
def handle_message(message):
    user_id   = message.from_user.id
    chat_id   = message.chat.id
    user_text = message.text.strip()

    logger.info(f"User {user_id}: {user_text[:80]}")

    # Show typing indicator
    send_typing(chat_id)

    # React to the user's message
    reaction_emoji = pick_reaction(user_text)
    try:
        bot.set_message_reaction(
            chat_id,
            message.message_id,
            [telebot.types.ReactionTypeEmoji(reaction_emoji)],
        )
    except Exception:
        pass  # Reactions not supported in all chat types

    # Get AI response
    raw_reply = get_ai_response(user_id, user_text)

    # Format for Telegram MarkdownV2
    formatted = escape_md(raw_reply)

    try:
        bot.reply_to(message, formatted, parse_mode="MarkdownV2")
    except Exception:
        # Fallback: plain text if markdown fails
        bot.reply_to(message, raw_reply)


@bot.message_handler(content_types=["photo", "document", "sticker", "voice"])
def handle_unsupported(message):
    bot.reply_to(
        message,
        "📝 I currently support *text messages* only\\. Please type your question\\!",
        parse_mode="MarkdownV2",
    )


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

    if RENDER_URL:
        # Webhook mode for Render.com deployment
        webhook_url = f"{RENDER_URL}/{BOT_TOKEN}"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook set → {webhook_url}")
        app.run(host="0.0.0.0", port=PORT)
    else:
        # Local polling mode
        logger.info("Starting in polling mode (local dev)...")
        bot.remove_webhook()
        threading.Thread(target=bot.infinity_polling, daemon=True).start()
        app.run(host="0.0.0.0", port=PORT)
