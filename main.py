import os
import logging
import random
import telebot
from telebot import types
from flask import Flask, request
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BOT_TOKEN  = os.environ.get("BOT_TOKEN")
NARA_TOKEN = os.environ.get("NARA_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

client = OpenAI(
    base_url="https://router.bynara.id/v1",
    api_key=NARA_TOKEN,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Startup check
logger.info(f"BOT_TOKEN loaded: {'YES' if BOT_TOKEN else 'NO - MISSING!'}")
logger.info(f"NARA_TOKEN loaded: {'YES' if NARA_TOKEN else 'NO - MISSING!'}")

# ─── PER-USER CONVERSATION MEMORY ─────────────────────────────────────────────
user_sessions: dict = {}

SYSTEM_PROMPT = """You are an intelligent, friendly, and helpful AI assistant inside Telegram.

Strict rules you must always follow:
1. NEVER use the # symbol anywhere. No hashtags, no markdown headers.
2. Do NOT start any line with #, ##, or ###.
3. Use *bold* (single asterisk on each side) for Telegram bold to highlight key points naturally.
4. Use emojis occasionally to feel warm, but don't overdo it.
5. Keep paragraphs short and readable. Break long answers into multiple short paragraphs.
6. Use plain numbers or dashes ( - ) for lists. Never use bullet symbols or header markers.
7. Be conversational, smart, and adapt to the user's tone.
8. For code blocks, use triple backticks with the language name.
9. Remember the conversation and give connected, coherent replies.
10. If asked who you are, say you're an AI assistant powered by DeepSeek.
"""

MAX_HISTORY = 20

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_history(user_id):
    return user_sessions.setdefault(user_id, [])

def trim_history(user_id):
    h = user_sessions.get(user_id, [])
    if len(h) > MAX_HISTORY:
        user_sessions[user_id] = h[-MAX_HISTORY:]

def add_to_history(user_id, role, content):
    get_history(user_id).append({"role": role, "content": content})
    trim_history(user_id)

def ask_deepseek(user_id, user_message):
    add_to_history(user_id, "user", user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + get_history(user_id)
    try:
        logger.info(f"Calling DeepSeek API for user {user_id}...")
        response = client.chat.completions.create(
            model="deepseek-3.2",
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        # Remove any accidental # headers
        lines = reply.split("\n")
        clean_lines = [l.lstrip("# ") if l.startswith("#") else l for l in lines]
        reply = "\n".join(clean_lines)
        add_to_history(user_id, "assistant", reply)
        logger.info(f"DeepSeek replied successfully for user {user_id}")
        return reply
    except Exception as e:
        logger.error(f"DeepSeek API ERROR for user {user_id}: {type(e).__name__}: {str(e)}")
        return (
            f"⚠️ AI se connect nahi ho paya.\n\n"
            f"Error: `{type(e).__name__}: {str(e)[:200]}`\n\n"
            "Render logs check karo ya NARA\\_TOKEN verify karo."
        )

def send_typing(chat_id):
    bot.send_chat_action(chat_id, "typing")

# ─── KEYBOARD ─────────────────────────────────────────────────────────────────

def main_menu():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add(
        types.KeyboardButton("💬 Chat with AI"),
        types.KeyboardButton("🔄 Clear History"),
        types.KeyboardButton("📊 My Stats"),
        types.KeyboardButton("ℹ️ About"),
    )
    return markup

# ─── REACTIONS ────────────────────────────────────────────────────────────────

REACTIONS = ["👍", "🔥", "❤️", "🤩", "👏", "💡", "✅", "🎯"]

def send_reaction(chat_id, message_id):
    try:
        emoji = random.choice(REACTIONS)
        bot.set_message_reaction(
            chat_id,
            message_id,
            reaction=[types.ReactionTypeEmoji(emoji)],
            is_big=False
        )
    except Exception:
        pass

# ─── COMMAND HANDLERS ─────────────────────────────────────────────────────────

@bot.message_handler(commands=["start"])
def start(message):
    name = message.from_user.first_name or "there"
    text = (
        f"👋 Hello *{name}*! I'm your AI-powered assistant.\n\n"
        "I'm connected to *DeepSeek AI* and ready to help you with anything — "
        "questions, coding, writing, analysis, or just a conversation.\n\n"
        "Just *type your message* to get started! 🚀"
    )
    bot.send_message(message.chat.id, text, parse_mode="Markdown", reply_markup=main_menu())


@bot.message_handler(commands=["help"])
def help_cmd(message):
    text = (
        "*Available Commands*\n\n"
        "/start — Welcome message\n"
        "/help — Show this help\n"
        "/clear — Clear conversation memory\n"
        "/stats — Your session statistics\n"
        "/about — About this bot\n"
        "/ping — Test if AI is reachable\n\n"
        "Or just type anything and I'll reply! 💬"
    )
    bot.send_message(message.chat.id, text, parse_mode="Markdown", reply_markup=main_menu())


@bot.message_handler(commands=["ping"])
def ping_cmd(message):
    """Test API connectivity"""
    bot.send_chat_action(message.chat.id, "typing")
    try:
        response = client.chat.completions.create(
            model="deepseek-3.2",
            messages=[{"role": "user", "content": "Say only: Pong!"}],
            max_tokens=10,
        )
        reply = response.choices[0].message.content.strip()
        bot.send_message(message.chat.id, f"✅ API connected! Response: {reply}")
    except Exception as e:
        bot.send_message(
            message.chat.id,
            f"❌ API connection failed!\n\nError: `{type(e).__name__}: {str(e)[:300]}`",
            parse_mode="Markdown"
        )


@bot.message_handler(commands=["clear"])
def clear_cmd(message):
    user_sessions[message.from_user.id] = []
    bot.send_message(
        message.chat.id,
        "🗑️ History cleared! Let's start fresh. What's on your mind?",
        reply_markup=main_menu()
    )


@bot.message_handler(commands=["stats"])
def stats_cmd(message):
    h = get_history(message.from_user.id)
    user_msgs = sum(1 for m in h if m["role"] == "user")
    bot_msgs  = sum(1 for m in h if m["role"] == "assistant")
    text = (
        f"📊 *Your Session Stats*\n\n"
        f"Messages you sent: *{user_msgs}*\n"
        f"My replies: *{bot_msgs}*\n"
        f"Total in memory: *{len(h)}*\n\n"
        "Use /clear to reset anytime."
    )
    bot.send_message(message.chat.id, text, parse_mode="Markdown")


@bot.message_handler(commands=["about"])
def about_cmd(message):
    text = (
        "*About This Bot*\n\n"
        "Powered by *DeepSeek 3.2* via the Nara AI router.\n\n"
        "- Remembers your conversation context\n"
        "- Clean, readable Telegram formatting\n"
        "- Handles coding, writing, Q&A and more\n"
        "- Runs 24/7 on Render.com\n\n"
        "Built with ❤️ using pyTelegramBotAPI."
    )
    bot.send_message(message.chat.id, text, parse_mode="Markdown")


# ─── BUTTON HANDLERS ──────────────────────────────────────────────────────────

@bot.message_handler(func=lambda m: m.text == "🔄 Clear History")
def btn_clear(message):
    clear_cmd(message)

@bot.message_handler(func=lambda m: m.text == "ℹ️ About")
def btn_about(message):
    about_cmd(message)

@bot.message_handler(func=lambda m: m.text == "📊 My Stats")
def btn_stats(message):
    stats_cmd(message)

@bot.message_handler(func=lambda m: m.text == "💬 Chat with AI")
def btn_chat(message):
    bot.send_message(
        message.chat.id,
        "💬 I'm listening! Type your message and I'll reply.",
        reply_markup=main_menu()
    )

# ─── MAIN MESSAGE HANDLER ─────────────────────────────────────────────────────

@bot.message_handler(func=lambda m: True, content_types=["text"])
def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    msg_id  = message.message_id

    send_reaction(chat_id, msg_id)
    send_typing(chat_id)

    reply = ask_deepseek(user_id, message.text.strip())
    bot.send_message(chat_id, reply, parse_mode="Markdown", reply_markup=main_menu())


# ─── FLASK WEBHOOK ────────────────────────────────────────────────────────────

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    json_data = request.get_json(force=True)
    update = telebot.types.Update.de_json(json_data)
    bot.process_new_updates([update])
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "Bot is alive! 🚀", 200

# ─── STARTUP ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")
    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/{BOT_TOKEN}"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook set: {webhook_url}")
    else:
        bot.remove_webhook()
        logger.info("Running in polling mode (local)...")
        bot.infinity_polling()
        import sys; sys.exit()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
