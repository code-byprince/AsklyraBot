import os
import logging
import telebot
from telebot import types
from flask import Flask, request
from openai import OpenAI
from datetime import datetime

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

# ─── PER-USER CONVERSATION MEMORY ─────────────────────────────────────────────
user_sessions: dict[int, list[dict]] = {}

SYSTEM_PROMPT = """You are an intelligent, friendly, and helpful AI assistant integrated into Telegram.

Rules you MUST follow strictly:
1. NEVER use the # symbol anywhere in your responses. Avoid hashtags completely.
2. NEVER use markdown headers (lines starting with #, ##, ###). Do not use them even if listing topics.
3. Use *bold* (single asterisks) for Telegram-style bold to highlight key points naturally within sentences.
4. Use emojis occasionally to make responses feel warm and engaging, but don't overdo it.
5. Keep responses concise and clear — break long answers into short readable paragraphs.
6. Use plain dashes ( - ) or numbers for lists, never bullet symbols or headers.
7. Be conversational, supportive, and adapt your tone to the user's style.
8. If asked about your identity, say you are an AI assistant powered by DeepSeek.
9. Remember context within the conversation to give coherent, connected replies.
10. For code, wrap it in triple backticks with the language name for Telegram's code formatting.
"""

MAX_HISTORY = 20  # messages to keep per user

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_history(user_id: int) -> list[dict]:
    return user_sessions.setdefault(user_id, [])


def trim_history(user_id: int):
    history = user_sessions.get(user_id, [])
    if len(history) > MAX_HISTORY:
        user_sessions[user_id] = history[-MAX_HISTORY:]


def add_to_history(user_id: int, role: str, content: str):
    get_history(user_id).append({"role": role, "content": content})
    trim_history(user_id)


def ask_deepseek(user_id: int, user_message: str) -> str:
    add_to_history(user_id, "user", user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + get_history(user_id)
    try:
        response = client.chat.completions.create(
            model="deepseek-3.2",
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        # Safety: strip any accidental # symbols
        reply = reply.replace("# ", "").replace("## ", "").replace("### ", "")
        add_to_history(user_id, "assistant", reply)
        return reply
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        return "⚠️ Sorry, I ran into an issue reaching the AI. Please try again in a moment."


def send_typing(chat_id: int):
    bot.send_chat_action(chat_id, "typing")


# ─── MAIN MENU KEYBOARD ───────────────────────────────────────────────────────

def main_menu():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add(
        types.KeyboardButton("🤖 Chat with AI"),
        types.KeyboardButton("🔄 Clear History"),
        types.KeyboardButton("ℹ️ About"),
        types.KeyboardButton("📊 My Stats"),
    )
    return markup


# ─── COMMAND HANDLERS ─────────────────────────────────────────────────────────

@bot.message_handler(commands=["start"])
def start(message):
    name = message.from_user.first_name or "there"
    welcome = (
        f"👋 Hello *{name}*! Welcome to your AI-powered assistant.\n\n"
        "I'm connected to *DeepSeek AI* and ready to help you with anything — "
        "questions, coding, writing, analysis, or just a friendly chat.\n\n"
        "Use the menu below or simply *type your message* to get started! 🚀"
    )
    bot.send_message(message.chat.id, welcome, parse_mode="Markdown", reply_markup=main_menu())


@bot.message_handler(commands=["help"])
def help_cmd(message):
    help_text = (
        "*Available Commands*\n\n"
        "/start — Welcome message & menu\n"
        "/help — Show this help\n"
        "/clear — Clear your conversation history\n"
        "/stats — See your session stats\n"
        "/about — About this bot\n\n"
        "You can also use the keyboard buttons below. Just type anything and I'll reply! 💬"
    )
    bot.send_message(message.chat.id, help_text, parse_mode="Markdown", reply_markup=main_menu())


@bot.message_handler(commands=["clear"])
def clear_history(message):
    user_id = message.from_user.id
    user_sessions[user_id] = []
    bot.send_message(
        message.chat.id,
        "🗑️ Conversation history cleared! We're starting fresh. What would you like to talk about?",
        reply_markup=main_menu()
    )


@bot.message_handler(commands=["stats"])
def stats_cmd(message):
    user_id = message.from_user.id
    history = get_history(user_id)
    user_msgs  = sum(1 for m in history if m["role"] == "user")
    bot_msgs   = sum(1 for m in history if m["role"] == "assistant")
    stats_text = (
        f"📊 *Your Session Stats*\n\n"
        f"Messages you sent: *{user_msgs}*\n"
        f"My replies: *{bot_msgs}*\n"
        f"Total exchanges: *{len(history)}*\n\n"
        f"Use /clear to reset the conversation anytime."
    )
    bot.send_message(message.chat.id, stats_text, parse_mode="Markdown")


@bot.message_handler(commands=["about"])
def about_cmd(message):
    about_text = (
        "*About This Bot*\n\n"
        "This is an advanced AI chatbot powered by *DeepSeek 3.2* via the Nara AI router.\n\n"
        "- Remembers your conversation context\n"
        "- Replies with clean, readable formatting\n"
        "- Handles coding, writing, Q&A, and more\n"
        "- Runs 24/7 on Render.com\n\n"
        "Built with ❤️ using pyTelegramBotAPI and OpenAI-compatible API."
    )
    bot.send_message(message.chat.id, about_text, parse_mode="Markdown")


# ─── KEYBOARD BUTTON HANDLERS ─────────────────────────────────────────────────

@bot.message_handler(func=lambda m: m.text == "🔄 Clear History")
def btn_clear(message):
    clear_history(message)


@bot.message_handler(func=lambda m: m.text == "ℹ️ About")
def btn_about(message):
    about_cmd(message)


@bot.message_handler(func=lambda m: m.text == "📊 My Stats")
def btn_stats(message):
    stats_cmd(message)


@bot.message_handler(func=lambda m: m.text == "🤖 Chat with AI")
def btn_chat(message):
    bot.send_message(
        message.chat.id,
        "💬 Great! Just type your message and I'll respond. Ask me anything!",
        reply_markup=main_menu()
    )


# ─── REACTION + AI REPLY ──────────────────────────────────────────────────────

REACTIONS = ["👍", "🔥", "❤️", "🤩", "👏", "💡", "✅", "🎯"]

import random

def send_reaction(chat_id: int, message_id: int):
    """Send a random emoji reaction to the user's message."""
    try:
        emoji = random.choice(REACTIONS)
        bot.set_message_reaction(
            chat_id,
            message_id,
            reaction=[types.ReactionTypeEmoji(emoji)],
            is_big=False
        )
    except Exception:
        pass  # Reactions may not be supported in all chat types — fail silently


@bot.message_handler(func=lambda m: True, content_types=["text"])
def handle_message(message):
    user_id   = message.from_user.id
    chat_id   = message.chat.id
    msg_id    = message.message_id
    user_text = message.text.strip()

    # React to the user's message
    send_reaction(chat_id, msg_id)

    # Show typing indicator
    send_typing(chat_id)

    # Get AI response
    reply = ask_deepseek(user_id, user_text)

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
    return "Bot is running! 🚀", 200


# ─── STARTUP ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")
    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/{BOT_TOKEN}"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook set: {webhook_url}")
    else:
        # Local polling fallback
        bot.remove_webhook()
        logger.info("Running in polling mode (local dev)...")
        bot.infinity_polling()
        import sys; sys.exit()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
