import os
import json
import logging
import sqlite3
import hashlib
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from flask import Flask
from telebot import TeleBot, types
from telebot.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
import openai
import requests
from io import BytesIO
import zipfile
import csv
import pandas as pd
from docx import Document
import PyPDF2
import tempfile
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment

# ============ CONFIGURATION ============
BOT_TOKEN = os.environ.get('BOT_TOKEN')
HF_TOKEN = os.environ.get('HF_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

if not BOT_TOKEN or not HF_TOKEN:
    raise ValueError("BOT_TOKEN and HF_TOKEN environment variables are required!")

# Initialize bot
bot = TeleBot(BOT_TOKEN, threaded=False)
app = Flask(__name__)

# ============ LOGGING ============
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============ DATABASE SETUP ============
def init_db():
    conn = sqlite3.connect('bot_database.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        first_name TEXT,
        last_name TEXT,
        join_date TIMESTAMP,
        last_active TIMESTAMP,
        total_messages INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        language TEXT DEFAULT 'en',
        is_admin INTEGER DEFAULT 0
    )''')
    
    # Messages table
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        message TEXT,
        response TEXT,
        tokens_used INTEGER,
        timestamp TIMESTAMP,
        response_time REAL
    )''')
    
    # User memories table
    c.execute('''CREATE TABLE IF NOT EXISTS user_memories (
        user_id INTEGER,
        key TEXT,
        value TEXT,
        timestamp TIMESTAMP,
        PRIMARY KEY (user_id, key)
    )''')
    
    # Errors table
    c.execute('''CREATE TABLE IF NOT EXISTS error_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        error TEXT,
        timestamp TIMESTAMP
    )''')
    
    # Chat history table for long-term memory
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        role TEXT,
        content TEXT,
        timestamp TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()

init_db()

# ============ DATABASE FUNCTIONS ============
def get_db():
    return sqlite3.connect('bot_database.db')

def update_user(user_id, username, first_name, last_name):
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO users 
                 (user_id, username, first_name, last_name, last_active)
                 VALUES (?, ?, ?, ?, ?)''',
              (user_id, username, first_name, last_name, datetime.now()))
    conn.commit()
    conn.close()

def get_user_language(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT language FROM users WHERE user_id = ?', (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 'en'

def set_user_language(user_id, language):
    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE users SET language = ? WHERE user_id = ?', (language, user_id))
    conn.commit()
    conn.close()

def increment_message_count(user_id, tokens=0, response_time=0):
    conn = get_db()
    c = conn.cursor()
    c.execute('''UPDATE users 
                 SET total_messages = total_messages + 1,
                     total_tokens = total_tokens + ?,
                     last_active = ?
                 WHERE user_id = ?''',
              (tokens, datetime.now(), user_id))
    conn.commit()
    conn.close()

def save_message(user_id, message, response, tokens_used, response_time):
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO messages 
                 (user_id, message, response, tokens_used, timestamp, response_time)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (user_id, message, response, tokens_used, datetime.now(), response_time))
    conn.commit()
    conn.close()
    
    # Save to chat history for long-term memory
    save_chat_history(user_id, 'user', message)
    save_chat_history(user_id, 'assistant', response)

def save_chat_history(user_id, role, content):
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO chat_history (user_id, role, content, timestamp)
                 VALUES (?, ?, ?, ?)''',
              (user_id, role, content, datetime.now()))
    conn.commit()
    conn.close()

def get_chat_history(user_id, limit=20):
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT role, content FROM chat_history 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC LIMIT ?''',
              (user_id, limit))
    results = c.fetchall()
    conn.close()
    return results[::-1]  # Return in chronological order

def get_user_context(user_id):
    """Get recent chat history and memories for context"""
    history = get_chat_history(user_id, 15)
    context = []
    for role, content in history:
        context.append({"role": role, "content": content})
    return context

def save_memory(user_id, key, value):
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO user_memories (user_id, key, value, timestamp)
                 VALUES (?, ?, ?, ?)''',
              (user_id, key, value, datetime.now()))
    conn.commit()
    conn.close()

def get_memory(user_id, key):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT value FROM user_memories WHERE user_id = ? AND key = ?', 
              (user_id, key))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def log_error(user_id, error):
    conn = get_db()
    c = conn.cursor()
    c.execute('INSERT INTO error_logs (user_id, error, timestamp) VALUES (?, ?, ?)',
              (user_id, str(error), datetime.now()))
    conn.commit()
    conn.close()

def get_stats():
    conn = get_db()
    c = conn.cursor()
    
    # Total users
    c.execute('SELECT COUNT(*) FROM users')
    total_users = c.fetchone()[0]
    
    # Active users (last 24 hours)
    c.execute('SELECT COUNT(*) FROM users WHERE last_active > ?', 
              (datetime.now() - timedelta(hours=24),))
    active_users = c.fetchone()[0]
    
    # Daily messages
    c.execute('SELECT COUNT(*) FROM messages WHERE timestamp > ?',
              (datetime.now() - timedelta(days=1),))
    daily_messages = c.fetchone()[0]
    
    # Total tokens
    c.execute('SELECT SUM(total_tokens) FROM users')
    total_tokens = c.fetchone()[0] or 0
    
    # Avg response time
    c.execute('SELECT AVG(response_time) FROM messages WHERE timestamp > ?',
              (datetime.now() - timedelta(days=7),))
    avg_response_time = c.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'total_users': total_users,
        'active_users': active_users,
        'daily_messages': daily_messages,
        'total_tokens': total_tokens,
        'avg_response_time': round(avg_response_time, 3)
    }

def get_leaderboard(limit=10):
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT username, total_messages, total_tokens 
                 FROM users 
                 ORDER BY total_messages DESC 
                 LIMIT ?''', (limit,))
    results = c.fetchall()
    conn.close()
    return results

# ============ AI FUNCTIONS ============
def get_ai_response(user_id, user_message, system_prompt=""):
    """Get response from DeepSeek AI with context"""
    try:
        # Get user context and memories
        context = get_user_context(user_id)
        
        # Build messages array
        messages = []
        
        # System prompt
        system = system_prompt or "You are a helpful AI assistant. Respond in the user's language. Format important points in **bold**. Do not use # or * symbols for formatting, only use **bold** for important points."
        messages.append({"role": "system", "content": system})
        
        # Add context from chat history
        if context:
            messages.extend(context[-10:])  # Last 10 messages for context
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Call DeepSeek API through HuggingFace
        client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
            timeout=120
        )
        
        start_time = time.time()
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V4-Pro:novita",
            messages=messages,
            max_tokens=4000,
            temperature=0.7
        )
        response_time = time.time() - start_time
        
        ai_message = response.choices[0].message.content
        
        # Format important points as bold (remove # and * if present)
        ai_message = format_response(ai_message)
        
        # Track usage
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
        
        # Save to database
        save_message(user_id, user_message, ai_message, tokens_used, response_time)
        increment_message_count(user_id, tokens_used, response_time)
        
        return ai_message
        
    except Exception as e:
        logger.error(f"AI Error: {str(e)}")
        log_error(user_id, str(e))
        return "⚠️ Sorry, I encountered an error. Please try again later."

def format_response(text):
    """Format response to use bold for important points, remove # and *"""
    # Remove # symbols
    text = text.replace('#', '')
    
    # Find important points and make them bold
    # Look for points that start with numbers, bullets, or key phrases
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Check if line starts with a number, bullet, or important keyword
        if re.match(r'^[\d]+[\.\)]', line.strip()) or \
           re.match(r'^[\•\-\*]', line.strip()) or \
           re.search(r'(important|key|main|note|summary|conclusion|result|finding|benefit|advantage|disadvantage|tip|warning|caution)', line.lower()):
            # Make the entire line bold
            formatted_lines.append(f"**{line}**")
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def summarize_text(text, max_length=500):
    """Summarize long text"""
    try:
        client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN
        )
        
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V4-Pro:novita",
            messages=[
                {"role": "system", "content": f"Summarize the following text in {max_length} characters or less. Focus on key points."},
                {"role": "user", "content": text[:10000]}  # Limit input
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except:
        return text[:max_length] + "..."

def translate_text(text, target_language):
    """Translate text to target language"""
    try:
        client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN
        )
        
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V4-Pro:novita",
            messages=[
                {"role": "system", "content": f"Translate the following text to {target_language}. Only return the translation."},
                {"role": "user", "content": text}
            ],
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except:
        return text

def detect_language(text):
    """Detect language of text"""
    try:
        client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN
        )
        
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V4-Pro:novita",
            messages=[
                {"role": "system", "content": "Detect the language of this text. Return only the language name in English."},
                {"role": "user", "content": text[:500]}
            ],
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
    except:
        return "unknown"

# ============ FILE PROCESSING FUNCTIONS ============
def process_pdf(file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return summarize_text(text)
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def process_docx(file_bytes):
    try:
        doc = Document(BytesIO(file_bytes))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return summarize_text(text)
    except Exception as e:
        return f"Error processing DOCX: {str(e)}"

def process_txt(file_bytes):
    try:
        text = file_bytes.decode('utf-8')
        return summarize_text(text)
    except Exception as e:
        return f"Error processing TXT: {str(e)}"

def process_csv(file_bytes):
    try:
        df = pd.read_csv(BytesIO(file_bytes))
        summary = f"CSV Analysis:\n"
        summary += f"Rows: {len(df)}\n"
        summary += f"Columns: {len(df.columns)}\n"
        summary += f"Column names: {', '.join(df.columns)}\n"
        summary += f"\nSummary statistics:\n{df.describe().to_string()}"
        return summary[:4000]
    except Exception as e:
        return f"Error processing CSV: {str(e)}"

def process_excel(file_bytes):
    try:
        df = pd.read_excel(BytesIO(file_bytes))
        summary = f"Excel Analysis:\n"
        summary += f"Rows: {len(df)}\n"
        summary += f"Columns: {len(df.columns)}\n"
        summary += f"Column names: {', '.join(df.columns)}\n"
        summary += f"\nSummary statistics:\n{df.describe().to_string()}"
        return summary[:4000]
    except Exception as e:
        return f"Error processing Excel: {str(e)}"

def process_json(file_bytes):
    try:
        data = json.loads(file_bytes.decode('utf-8'))
        return json.dumps(data, indent=2, ensure_ascii=False)[:4000]
    except Exception as e:
        return f"Error processing JSON: {str(e)}"

def process_zip(file_bytes):
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as zip_file:
            files = zip_file.namelist()
            summary = f"ZIP Archive contains {len(files)} files:\n"
            for file in files[:20]:  # Limit to 20 files
                summary += f"- {file}\n"
            return summary
    except Exception as e:
        return f"Error processing ZIP: {str(e)}"

def process_code_file(file_bytes, filename):
    try:
        text = file_bytes.decode('utf-8')
        # Detect language from extension
        ext = filename.split('.')[-1].lower()
        language_map = {
            'py': 'Python',
            'js': 'JavaScript',
            'java': 'Java',
            'cpp': 'C++',
            'c': 'C',
            'cs': 'C#',
            'go': 'Go',
            'rs': 'Rust',
            'rb': 'Ruby',
            'php': 'PHP',
            'html': 'HTML',
            'css': 'CSS',
            'json': 'JSON',
            'xml': 'XML',
            'sql': 'SQL'
        }
        lang = language_map.get(ext, 'Code')
        
        summary = f"**{lang} Code Analysis:**\n"
        summary += f"Lines: {len(text.splitlines())}\n"
        summary += f"Characters: {len(text)}\n"
        
        # Try to analyze with AI
        try:
            analysis = get_ai_response(0, f"Analyze this {lang} code and summarize its purpose and structure:\n\n{text[:3000]}")
            summary += f"\n**AI Analysis:**\n{analysis}"
        except:
            summary += f"\nCode preview:\n{text[:1000]}..."
        
        return summary[:4000]
    except Exception as e:
        return f"Error processing code file: {str(e)}"

# ============ VOICE PROCESSING ============
def process_voice(file_bytes):
    try:
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as temp_audio:
            temp_audio.write(file_bytes)
            temp_audio_path = temp_audio.name
        
        # Convert to WAV for speech recognition
        audio = AudioSegment.from_ogg(temp_audio_path)
        wav_path = temp_audio_path.replace('.ogg', '.wav')
        audio.export(wav_path, format='wav')
        
        # Recognize speech
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Cleanup temp files
        os.unlink(temp_audio_path)
        os.unlink(wav_path)
        
        return text
    except Exception as e:
        return f"Error processing voice: {str(e)}"

# ============ BOT COMMANDS ============
@bot.message_handler(commands=['start'])
def handle_start(message):
    user = message.from_user
    update_user(user.id, user.username, user.first_name, user.last_name)
    
    welcome_msg = """**🤖 Welcome to Advanced AI Bot!**

I'm your intelligent assistant powered by DeepSeek AI. Here's what I can do:

**📚 File Processing:**
• 📄 PDF Summarize
• 📝 DOCX Reading
• 📃 TXT Reading
• 📊 CSV Analysis
• 📈 Excel Analysis
• 📋 JSON Formatter
• 📦 ZIP Extraction
• 💻 Code Analysis

**🎤 Voice & Translation:**
• Voice Message Support
• Auto Language Detection
• 100+ Language Translation

**🧠 Advanced Features:**
• Long-term Memory
• Conversation Summary
• Continue Response Button
• Admin Tools

**📊 Stats & Analytics:**
• Total Users
• Active Users
• Daily Messages
• Token Usage

**Commands:**
/help - Show all commands
/stats - View bot statistics
/leaderboard - Top users
/memory - View your memories
/summary - Summarize current conversation
/translate [lang] - Translate last message
/continue - Continue last response

Start chatting with me now! 🚀"""
    bot.reply_to(message, welcome_msg, parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def handle_help(message):
    help_msg = """**📋 Available Commands:**

**General:**
/start - Welcome message
/help - Show this help menu
/stats - Bot statistics
/leaderboard - Top users

**Memory & Context:**
/memory - View your saved memories
/summary - Summarize current conversation
/continue - Continue last response

**Translation:**
/translate [language] - Translate last message
/language [lang] - Set your language preference

**File Processing:**
Send any file (PDF, DOCX, TXT, CSV, Excel, JSON, ZIP, Code files)
Voice messages are automatically transcribed

**Admin Commands:**
/admin - Admin panel
/broadcast - Send message to all users
/stats_full - Detailed statistics

💡 **Pro Tips:**
• Send long messages for summaries
• Share files for automatic analysis
• The bot remembers your conversation
• Important points are shown in **bold**

Need help? Just ask! 🤖"""
    bot.reply_to(message, help_msg, parse_mode='Markdown')

@bot.message_handler(commands=['stats'])
def handle_stats(message):
    stats = get_stats()
    stats_msg = f"""**📊 Bot Statistics:**

👥 **Total Users:** {stats['total_users']}
🟢 **Active Users (24h):** {stats['active_users']}
💬 **Daily Messages:** {stats['daily_messages']}
🔢 **Total Tokens Used:** {stats['total_tokens']:,}
⚡ **Avg Response Time:** {stats['avg_response_time']}s

🤖 **AI Model:** DeepSeek V4 Pro
📅 **Status:** 🟢 Online"""
    bot.reply_to(message, stats_msg, parse_mode='Markdown')

@bot.message_handler(commands=['leaderboard'])
def handle_leaderboard(message):
    users = get_leaderboard(10)
    if not users:
        bot.reply_to(message, "No users found.")
        return
    
    board = "**🏆 Top Users Leaderboard:**\n\n"
    for i, (username, msgs, tokens) in enumerate(users, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        username = username or "Anonymous"
        board += f"{medal} **{username}** - {msgs} messages ({tokens} tokens)\n"
    
    bot.reply_to(message, board, parse_mode='Markdown')

@bot.message_handler(commands=['memory'])
def handle_memory(message):
    user_id = message.from_user.id
    
    # Get user's memories
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT key, value FROM user_memories WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10', (user_id,))
    memories = c.fetchall()
    conn.close()
    
    if not memories:
        bot.reply_to(message, "You don't have any saved memories yet.")
        return
    
    msg = "**🧠 Your Memories:**\n\n"
    for key, value in memories:
        msg += f"• **{key}:** {value[:100]}\n"
    
    bot.reply_to(message, msg, parse_mode='Markdown')

@bot.message_handler(commands=['summary'])
def handle_summary(message):
    user_id = message.from_user.id
    history = get_chat_history(user_id, 20)
    
    if len(history) < 3:
        bot.reply_to(message, "Not enough conversation history to summarize.")
        return
    
    # Create summary
    conversation = "\n".join([f"{role}: {content[:200]}" for role, content in history])
    summary = get_ai_response(user_id, f"Summarize this conversation concisely:\n{conversation}", 
                              "Provide a brief summary of the key points discussed.")
    
    bot.reply_to(message, f"**📝 Conversation Summary:**\n\n{summary}", parse_mode='Markdown')

@bot.message_handler(commands=['translate'])
def handle_translate(message):
    user_id = message.from_user.id
    
    # Get target language from command
    parts = message.text.split(' ', 1)
    if len(parts) < 2:
        bot.reply_to(message, "Please specify a language. Example: /translate Hindi")
        return
    
    target_lang = parts[1].strip()
    
    # Get last user message
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT message FROM messages 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC LIMIT 1''', (user_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        bot.reply_to(message, "No previous messages found to translate.")
        return
    
    original = result[0]
    translated = translate_text(original, target_lang)
    
    bot.reply_to(message, f"**Translated to {target_lang}:**\n\n{translated}", parse_mode='Markdown')

@bot.message_handler(commands=['language'])
def handle_language(message):
    parts = message.text.split(' ', 1)
    if len(parts) < 2:
        bot.reply_to(message, "Please specify a language code. Example: /language hi")
        return
    
    lang_code = parts[1].strip()
    set_user_language(message.from_user.id, lang_code)
    bot.reply_to(message, f"✅ Language set to: {lang_code}")

@bot.message_handler(commands=['continue'])
def handle_continue(message):
    user_id = message.from_user.id
    
    # Get last response
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT response FROM messages 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC LIMIT 1''', (user_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        bot.reply_to(message, "No previous response to continue.")
        return
    
    # Request continuation
    continuation = get_ai_response(user_id, "Please continue your previous response or elaborate further.",
                                   "Continue the previous response naturally. Don't repeat the entire previous message.")
    
    bot.reply_to(message, f"**📝 Continued Response:**\n\n{continuation}", parse_mode='Markdown')

# ============ FILE HANDLING ============
@bot.message_handler(content_types=['document'])
def handle_document(message):
    user_id = message.from_user.id
    file_info = message.document
    file_name = file_info.file_name
    file_id = file_info.file_id
    
    # Get file
    file = bot.get_file(file_id)
    file_bytes = bot.download_file(file.file_path)
    
    # Process based on file type
    ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
    
    processing_msg = bot.reply_to(message, f"📂 Processing {file_name}...")
    
    try:
        if ext == 'pdf':
            result = process_pdf(file_bytes)
        elif ext in ['docx', 'doc']:
            result = process_docx(file_bytes)
        elif ext in ['txt', 'log']:
            result = process_txt(file_bytes)
        elif ext == 'csv':
            result = process_csv(file_bytes)
        elif ext in ['xlsx', 'xls']:
            result = process_excel(file_bytes)
        elif ext == 'json':
            result = process_json(file_bytes)
        elif ext == 'zip':
            result = process_zip(file_bytes)
        elif ext in ['py', 'js', 'java', 'cpp', 'c', 'cs', 'go', 'rs', 'rb', 'php', 'html', 'css', 'sql']:
            result = process_code_file(file_bytes, file_name)
        else:
            result = "Unsupported file type. Please send PDF, DOCX, TXT, CSV, Excel, JSON, ZIP, or code files."
        
        bot.edit_message_text(f"**📄 {file_name} Analysis:**\n\n{result[:4000]}", 
                             chat_id=message.chat.id,
                             message_id=processing_msg.message_id,
                             parse_mode='Markdown')
                             
    except Exception as e:
        bot.edit_message_text(f"❌ Error processing file: {str(e)}", 
                             chat_id=message.chat.id,
                             message_id=processing_msg.message_id)

@bot.message_handler(content_types=['voice', 'audio'])
def handle_voice(message):
    user_id = message.from_user.id
    
    # Get voice file
    if message.voice:
        file_id = message.voice.file_id
    else:
        file_id = message.audio.file_id
    
    file = bot.get_file(file_id)
    file_bytes = bot.download_file(file.file_path)
    
    processing_msg = bot.reply_to(message, "🎤 Processing voice message...")
    
    try:
        text = process_voice(file_bytes)
        bot.edit_message_text(f"**📝 Transcribed Text:**\n\n{text}", 
                             chat_id=message.chat.id,
                             message_id=processing_msg.message_id,
                             parse_mode='Markdown')
        
        # Auto-detect language
        lang = detect_language(text)
        if lang and lang.lower() != 'unknown':
            set_user_language(user_id, lang)
        
        # Get AI response
        response = get_ai_response(user_id, text)
        bot.send_message(message.chat.id, f"**🤖 Response:**\n\n{response}", parse_mode='Markdown')
        
    except Exception as e:
        bot.edit_message_text(f"❌ Error processing voice: {str(e)}", 
                             chat_id=message.chat.id,
                             message_id=processing_msg.message_id)

# ============ GROUP ADMIN TOOLS ============
@bot.message_handler(commands=['admin'], func=lambda msg: msg.from_user.id in [int(os.environ.get('ADMIN_IDS', '0').split(','))])
def admin_panel(message):
    stats = get_stats()
    
    panel = f"""**🔐 Admin Panel**

**📊 Stats:**
• Total Users: {stats['total_users']}
• Active Users: {stats['active_users']}
• Daily Messages: {stats['daily_messages']}
• Total Tokens: {stats['total_tokens']:,}
• Avg Response: {stats['avg_response_time']}s

**🛠 Commands:**
/broadcast [message] - Send to all users
/stats_full - Detailed stats
/error_logs - View errors
/clear_errors - Clear errors
/user_info [id] - User details
/ban [id] - Ban user
/unban [id] - Unban user
"""
    bot.reply_to(message, panel, parse_mode='Markdown')

@bot.message_handler(commands=['broadcast'], func=lambda msg: msg.from_user.id in [int(os.environ.get('ADMIN_IDS', '0').split(','))])
def broadcast(message):
    parts = message.text.split(' ', 1)
    if len(parts) < 2:
        bot.reply_to(message, "Please provide a message. Example: /broadcast Hello everyone!")
        return
    
    msg = parts[1]
    
    # Get all users
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT user_id FROM users')
    users = c.fetchall()
    conn.close()
    
    success = 0
    fail = 0
    
    bot.reply_to(message, f"📢 Broadcasting to {len(users)} users...")
    
    for user in users:
        try:
            bot.send_message(user[0], f"📢 **Announcement:**\n\n{msg}", parse_mode='Markdown')
            success += 1
        except:
            fail += 1
        time.sleep(0.1)  # Rate limit
    
    bot.send_message(message.chat.id, f"✅ Broadcast complete!\nSuccess: {success}\nFailed: {fail}")

@bot.message_handler(commands=['stats_full'], func=lambda msg: msg.from_user.id in [int(os.environ.get('ADMIN_IDS', '0').split(','))])
def stats_full(message):
    conn = get_db()
    c = conn.cursor()
    
    # Detailed stats
    c.execute('SELECT COUNT(*) FROM users')
    total = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM users WHERE last_active > datetime("now", "-7 days")')
    weekly = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM messages')
    total_msgs = c.fetchone()[0]
    
    c.execute('SELECT SUM(tokens_used) FROM messages')
    total_tokens = c.fetchone()[0] or 0
    
    c.execute('SELECT COUNT(*) FROM error_logs')
    errors = c.fetchone()[0]
    
    c.execute('SELECT AVG(response_time) FROM messages')
    avg_time = c.fetchone()[0] or 0
    
    conn.close()
    
    stats_msg = f"""**📊 Full Statistics:**

**Users:**
• Total: {total}
• Active (7d): {weekly}
• Active Rate: {round(weekly/total*100, 1)}%

**Messages:**
• Total: {total_msgs:,}
• Per User Avg: {round(total_msgs/total, 1) if total > 0 else 0}

**AI Usage:**
• Total Tokens: {total_tokens:,}
• Avg Response: {round(avg_time, 3)}s
• Errors: {errors}

**Performance:**
• Status: 🟢 Online
• Model: DeepSeek V4 Pro
"""
    bot.reply_to(message, stats_msg, parse_mode='Markdown')

# ============ MAIN MESSAGE HANDLER ============
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user = message.from_user
    user_id = user.id
    text = message.text
    
    # Skip commands
    if text.startswith('/'):
        return
    
    # Update user
    update_user(user_id, user.username, user.first_name, user.last_name)
    
    # Check for long message
    if len(text) > 500:
        response_text = get_ai_response(user_id, f"Summarize this and respond:\n{text}")
    else:
        response_text = get_ai_response(user_id, text)
    
    # Auto-detect language
    lang = detect_language(text)
    if lang and lang.lower() != 'unknown':
        set_user_language(user_id, lang)
    
    # Send response with continue button
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("🔄 Continue", callback_data="continue"))
    markup.add(InlineKeyboardButton("📝 Summarize", callback_data="summarize"))
    
    bot.send_message(message.chat.id, response_text, parse_mode='Markdown', reply_markup=markup)

# ============ CALLBACK HANDLERS ============
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    user_id = call.from_user.id
    
    if call.data == "continue":
        # Get last response and continue
        conn = get_db()
        c = conn.cursor()
        c.execute('''SELECT response FROM messages 
                     WHERE user_id = ? 
                     ORDER BY timestamp DESC LIMIT 1''', (user_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            continuation = get_ai_response(user_id, "Please continue the previous response or elaborate further.",
                                          "Continue the previous response naturally. Don't repeat the entire previous message.")
            bot.send_message(call.message.chat.id, f"**📝 Continued:**\n\n{continuation}", parse_mode='Markdown')
        else:
            bot.answer_callback_query(call.id, "No previous response found.")
    
    elif call.data == "summarize":
        # Get conversation summary
        history = get_chat_history(user_id, 20)
        if len(history) >= 3:
            conv = "\n".join([f"{role}: {content[:200]}" for role, content in history])
            summary = get_ai_response(user_id, f"Summarize this conversation:\n{conv}",
                                     "Provide a concise summary of the key points.")
            bot.send_message(call.message.chat.id, f"**📝 Summary:**\n\n{summary}", parse_mode='Markdown')
        else:
            bot.answer_callback_query(call.id, "Not enough conversation history to summarize.")
    
    elif call.data == "translate":
        # Get last message and translate
        conn = get_db()
        c = conn.cursor()
        c.execute('''SELECT message FROM messages 
                     WHERE user_id = ? 
                     ORDER BY timestamp DESC LIMIT 1''', (user_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            lang = get_user_language(user_id)
            translated = translate_text(result[0], lang)
            bot.send_message(call.message.chat.id, f"**Translated to {lang}:**\n\n{translated}", parse_mode='Markdown')
        else:
            bot.answer_callback_query(call.id, "No message to translate.")
    
    bot.answer_callback_query(call.id)

# ============ WEBHOOK SETUP ============
@app.route('/webhook', methods=['POST'])
def webhook():
    import flask
    if flask.request.headers.get('content-type') == 'application/json':
        json_string = flask.request.get_data().decode('utf-8')
        update = types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        flask.abort(403)

@app.route('/')
def home():
    return "Bot is running!"

# ============ MAIN ============
if __name__ == '__main__':
    # Remove webhook if set
    bot.remove_webhook()
    
    # Set webhook
    import os
    webhook_url = os.environ.get('RENDER_EXTERNAL_URL', 'https://your-bot-url.com')
    if webhook_url != 'https://your-bot-url.com':
        bot.set_webhook(url=f"{webhook_url}/webhook")
        print(f"✅ Webhook set to: {webhook_url}/webhook")
    else:
        print("⚠️ Webhook URL not set. Use polling mode.")
        bot.polling(none_stop=True)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
