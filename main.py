import os
import telebot
from openai import OpenAI
from flask import Flask
from threading import Thread

# 1. Environment Variables
BOT_TOKEN = os.environ.get('BOT_TOKEN')
HF_TOKEN = os.environ.get('HF_TOKEN')

# 2. Initialize OpenAI Client (Hugging Face Router)
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# 3. Initialize Telegram Bot
bot = telebot.TeleBot(BOT_TOKEN)

# 4. Initialize Flask App (Required for Render)
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive!"

# 5. Telegram Message Handler
@bot.message_handler(func=lambda message: True)
def chat_with_ai(message):
    try:
        # Show "typing..." status in Telegram
        bot.send_chat_action(message.chat.id, 'typing')

        # Call Hugging Face API
        chat_completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3", # You can use the model string provided or "deepseek-ai/DeepSeek-V3"
            messages=[
                {
                    "role": "user",
                    "content": message.text,
                },
            ],
        )

        # Get response content
        ai_response = chat_completion.choices[0].message.content
        
        # Send response back to Telegram
        bot.reply_to(message, ai_response)

    except Exception as e:
        print(f"Error: {e}")
        bot.reply_to(message, "Sorry, I encountered an error processing that request.")

# 6. Functions to run the bot and server together
def run_bot():
    bot.polling(none_stop=True)

if __name__ == "__main__":
    # Start the Telegram bot in a separate thread
    Thread(target=run_bot).start()
    
    # Start the Flask server on the port provided by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
