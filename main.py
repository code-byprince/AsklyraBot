import os
import time
import logging
import telebot
from telebot import types
from openai import OpenAI
from flask import Flask, request

# ── Logging Setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Environment Variables ───────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN")
HF_TOKEN  = os.environ.get("HF_TOKEN")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is not set!")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set!")

# ── Client Setup ────────────────────────────────────────────────
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

ai_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# ── Flask App for Render.com ────────────────────────────────────
app = Flask(__name__)

# ── In-memory Conversation History (per user) ──────────────────
conversation_history = {}
MAX_HISTORY = 10  # last N exchanges kept

SYSTEM_PROMPT = """You are a smart, helpful, and friendly AI assistant on Telegram.

Rules you must ALWAYS follow:
- Never use markdown headers like # or ## or ###
- Never use hashtags
- Use plain text formatting only
- Use bold text by wrapping key phrases with <b>...</b> (HTML)
- Use italic with <i>...</i> for emphasis
- Use line breaks for structure, not headers
- Keep responses concise but informative
- Be warm, natural, and conversational
- When listing items, use simple dashes (-) or numbers
- Highlight the most important point in every response using <b>bold</b>
- React to the user's emotional tone — if they seem happy, be warm; if stuck, be encouraging"""

# ── Reaction Emojis based on message content ───────────────────
def pick_reaction(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["thank", "shukriya", "thanks", "dhanyawad"]):
        return "🙏"
    if any(w in text_lower for w in ["hello", "hi", "hey", "namaste", "hii"]):
        return "👋"
    if any(w in text_lower for w in ["help", "problem", "issue", "error", "stuck"]):
        return "🛠️"
    if any(w in text_lower for w in ["joke", "funny", "haha", "lol", "mazak"]):
        return "😂"
    if any(w in text_lower for w in ["sad", "cry", "upset", "dukhi", "bura"]):
        return "❤️"
    if any(w in text_lower for w in ["wow", "amazing", "awesome", "great", "zabardast"]):
        return "🔥"
    if any(w in text_lower for w in ["code", "program", "python", "javascript", "script"]):
        return "💻"
    if any(w in text_lower for w in ["idea", "suggest", "plan", "soch"]):
        return "💡"
    return "✨"

# ── Send Reaction (Telegram emoji reaction) ────────────────────
def send_reaction(chat_id: int, message_id: int, emoji: str):
    try:
        bot.set_message_reaction(
            chat_id=chat_id,
            message_id=message_id,
            reaction=[types.ReactionTypeEmoji(emoji=emoji)],
            is_big=False
        )
    except Exception as e:
        logger.warning(f"Reaction failed (may not be supported in this chat type): {e}")

# ── Call DeepSeek AI ───────────────────────────────────────────
def ask_deepseek(user_id: int, user_message: str) -> str:
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    conversation_history[user_id].append({
        "role": "user",
        "content": user_message
    })

    # Keep history trimmed
    if len(conversation_history[user_id]) > MAX_HISTORY * 2:
        conversation_history[user_id] = conversation_history[user_id][-(MAX_HISTORY * 2):]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[user_id]

    try:
        response = ai_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324:novita",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()

        # Remove any accidental # headings AI might generate
        lines = reply.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                # Convert heading to bold text instead
                text = stripped.lstrip("#").strip()
                cleaned.append(f"<b>{text}</b>")
            else:
                cleaned.append(line)
        reply = "\n".join(cleaned)

        conversation_history[user_id].append({
            "role": "assistant",
            "content": reply
        })

        return reply

    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        return "⚠️ <b>Oops!</b> AI se connect nahi ho paya. Thodi der baad try karo."

# ── /start Command ─────────────────────────────────────────────
@bot.message_handler(commands=["start"])
def handle_start(message):
    user_name = message.from_user.first_name or "dost"
    send_reaction(message.chat.id, message.message_id, "👋")

    welcome = (
        f"👋 <b>Namaste, {user_name}!</b>\n\n"
        "Main hoon tera <b>AI Assistant</b> — powered by <b>DeepSeek</b> 🤖\n\n"
        "Tum mujhse kuch bhi pooch sakte ho:\n"
        "- 💡 Ideas aur suggestions\n"
        "- 💻 Coding help\n"
        "- 📚 Koi bhi topic explain karna\n"
        "- 🗣️ Normal baatein bhi!\n\n"
        "<b>Bas likho — main hoon yahan!</b>"
    )
    bot.reply_to(message, welcome)

# ── /help Command ──────────────────────────────────────────────
@bot.message_handler(commands=["help"])
def handle_help(message):
    send_reaction(message.chat.id, message.message_id, "🛠️")

    help_text = (
        "🛠️ <b>Commands List</b>\n\n"
        "/start — Bot shuru karo\n"
        "/help — Yeh menu dekho\n"
        "/clear — Apni conversation history delete karo\n"
        "/about — Bot ke baare mein jano\n\n"
        "<b>Tip:</b> Seedha apna sawaal likho — main samajh lunga! 😊"
    )
    bot.reply_to(message, help_text)

# ── /clear Command ─────────────────────────────────────────────
@bot.message_handler(commands=["clear"])
def handle_clear(message):
    user_id = message.from_user.id
    conversation_history.pop(user_id, None)
    send_reaction(message.chat.id, message.message_id, "✨")
    bot.reply_to(message, "🗑️ <b>Conversation history clear ho gayi!</b>\nFresh start karo. 😊")

# ── /about Command ─────────────────────────────────────────────
@bot.message_handler(commands=["about"])
def handle_about(message):
    send_reaction(message.chat.id, message.message_id, "💡")
    about_text = (
        "🤖 <b>About This Bot</b>\n\n"
        "Model: <b>DeepSeek-V3</b> via HuggingFace Router\n"
        "Memory: Last 10 messages yaad rehte hain\n"
        "Language: Hindi + English dono samajhta hoon\n\n"
        "<b>Built with:</b> Python, pyTelegramBotAPI, OpenAI SDK"
    )
    bot.reply_to(message, about_text)

# ── Main Message Handler ────────────────────────────────────────
@bot.message_handler(func=lambda msg: True, content_types=["text"])
def handle_message(message):
    user_id   = message.from_user.id
    user_text = message.text.strip()

    if not user_text:
        return

    # Send reaction first based on user's message
    emoji = pick_reaction(user_text)
    send_reaction(message.chat.id, message.message_id, emoji)

    # Show typing indicator
    bot.send_chat_action(message.chat.id, "typing")

    # Get AI response
    reply = ask_deepseek(user_id, user_text)

    bot.reply_to(message, reply)

# ── Webhook Route for Render.com ───────────────────────────────
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")  # optional: set on Render

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        json_data = request.get_data(as_text=True)
        update = telebot.types.Update.de_json(json_data)
        bot.process_new_updates([update])
        return "OK", 200
    return "Bad Request", 400

@app.route("/", methods=["GET"])
def index():
    return "✅ Bot is running!", 200

# ── Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    if WEBHOOK_URL:
        # Webhook mode (for Render.com)
        bot.remove_webhook()
        time.sleep(1)
        bot.set_webhook(url=f"{WEBHOOK_URL}/{BOT_TOKEN}")
        logger.info(f"Webhook set: {WEBHOOK_URL}/{BOT_TOKEN}")
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        # Polling mode (for local testing)
        logger.info("Starting in polling mode...")
        bot.remove_webhook()
        bot.infinity_polling(timeout=30, long_polling_timeout=20)
