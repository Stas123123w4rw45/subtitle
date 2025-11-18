# -*- coding: utf-8 -*-
import os
import sys
import threading
import subprocess
import tempfile
import time
import shutil
import logging
import asyncio
import re
from groq import Groq 

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes,
    ConversationHandler, CallbackQueryHandler
)
from http.server import HTTPServer, BaseHTTPRequestHandler

# Налаштування логування
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)

# --- ВАШІ НАЛАШТУВАННЯ ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_CRF = "28" 
WORDS_PER_LINE = 2 
MAX_LINES_PER_PAGE = 1 
# -------------------------

# --- Стани для діалогу ---
STATE_RECEIVE_VIDEO, STATE_RECEIVE_EDIT = range(2)

# --- Функції-хелпери ---

def find_ffmpeg():
    # Спрощена логіка: спочатку шукаємо в PATH, потім локально
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    local_path = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.isfile(local_path):
        return local_path
    return "ffmpeg"

def escape_for_subtitles_filter(path):
    p = os.path.abspath(path)
    if os.name == 'nt':
        p = p.replace("\\", "/").replace(":", "\\:")
    return p

def ass_time(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60); cs = int(round((sec - int(sec))*100))
    return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

def escape_ass_text(txt: str) -> str:
    return txt.replace("{", r"\{").replace("}", r"\}")

# --- [ЛОГІКА GROQ API] ---

def transcribe_with_groq(file_path):
    if not GROQ_API_KEY:
        raise ValueError("Не вказано GROQ_API_KEY!")

    client = Groq(api_key=GROQ_API_KEY)
    
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="uk"
        )

    all_words = []
    full_text = transcription.text
    
    if hasattr(transcription, 'words'):
        for w in transcription.words:
            all_words.append((float(w['start']), float(w['end']), w['word'].strip()))
        
    return full_text, all_words

# --- [ЛОГІКА СИНХРОНІЗАЦІЇ] ---

def norm_token(t: str) -> str:
    return re.sub(r"[^a-zA-Zа-яА-Я0-9іїєґІЇЄҐ']+", "", t).lower()

def _parse_transcript_for_tokens(raw_text):
    tokens = []
    manual_starts = set()
    i = 0
    n = len(raw_text)
    next_is_line_start = False
    while i < n:
        ch = raw_text[i]
        if ch.isspace():
            j = i
            while j < n and raw_text[j].isspace(): j += 1
            if '  ' in raw_text[i:j] or '\n' in raw_text[i:j]:
                next_is_line_start = True
            i = j
            continue
        j = i
        while j < n and (not raw_text[j].isspace()): j += 1
        word = raw_text[i:j]
        if word:
            tokens.append(word)
            if next_is_line_start:
                manual_starts.add(len(tokens)-1)
                next_is_line_start = False
        i = j
    return tokens, manual_starts

def _align_tokens(src_word_segments, tgt_tokens):
    if not src_word_segments: return []
    src_words = [(float(t0), float(t1), w) for (t0,t1,w) in src_word_segments]
    src_norm = [norm_token(w) for (_,_,w) in src_words]
    tgt_norm = [norm_token(t) for t in tgt_tokens]
    n, m = len(src_norm), len(tgt_norm)
    if m == 0: return []
    if n == 0: return [(0.0, 0.0)] * m

    dp = [[(0, 'S')] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = (i, 'D')
    for j in range(1, m + 1): dp[0][j] = (j, 'I')
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if (src_norm[i-1] == tgt_norm[j-1] and src_norm[i-1]) else 1
            sub_cost = dp[i-1][j-1][0] + cost
            del_cost = dp[i-1][j][0] + 1
            ins_cost = dp[i][j-1][0] + 1
            if sub_cost <= del_cost and sub_cost <= ins_cost: dp[i][j] = (sub_cost, 'M')
            elif del_cost < ins_cost: dp[i][j] = (del_cost, 'D')
            else: dp[i][j] = (ins_cost, 'I')

    mapping = [-1] * m
    i, j = n, m
    while i > 0 or j > 0:
        op = dp[i][j][1]
        if op == 'M':
            if j > 0: mapping[j-1] = i-1
            i -= 1; j -= 1
        elif op == 'D': i -= 1
        elif op == 'I':
            if j > 0: mapping[j-1] = max(0, i - 1)
            j -= 1
        else: break
            
    token_times = []
    for j in range(m):
        mi = mapping[j]
        if mi >= 0 and mi < len(src_words):
            token_times.append((src_words[mi][0], src_words[mi][1]))
        else:
            ref_i = -1
            if j > 0 and mapping[j-1] >= 0: ref_i = mapping[j-1]
            elif j < m - 1 and mapping[j+1] >= 0: ref_i = mapping[j+1]
            elif j > 0: ref_i = mapping[j-1]
            if ref_i >= 0 and ref_i < len(src_words):
                t_ref = (src_words[ref_i][0] + src_words[ref_i][1]) / 2.0
                token_times.append((t_ref, t_ref + 0.1))
            else: token_times.append((0.0, 0.1))
    return token_times

def _pages_from_manual_or_auto(total_tokens, manual_starts, wpl=2, max_lines=1):
    if total_tokens <= 0: return []
    starts = sorted({s for s in manual_starts if 0 < s < total_tokens})
    lines = []
    if starts:
        all_starts = [0] + starts
        for si, s in enumerate(all_starts):
            seg_start = s
            seg_end = all_starts[si+1] if si+1 < len(all_starts) else total_tokens
            cur = seg_start
            while cur < seg_end:
                ln_end = min(cur + max(1, int(wpl)), seg_end)
                lines.append(list(range(cur, ln_end)))
                cur = ln_end
    else:
        cur = 0
        while cur < total_tokens:
            ln_end = min(cur + max(1, int(wpl)), total_tokens)
            lines.append(list(range(cur, ln_end)))
            cur = ln_end
    ml = max(1, int(max_lines))
    pages = []
    cur = 0
    while cur < len(lines):
        pages.append(lines[cur:cur+ml])
        cur += ml
    return pages

def _events_from_layout(tokens, token_times, manual_starts):
    n = len(tokens)
    if n == 0 or not token_times or len(token_times) != n: return []
    mids = []
    for j in range(n - 1):
        c1 = (token_times[j][0] + token_times[j][1]) / 2.0
        c2 = (token_times[j+1][0] + token_times[j+1][1]) / 2.0
        mids.append((c1 + c2) / 2.0)
    
    pages = _pages_from_manual_or_auto(len(tokens), manual_starts, WORDS_PER_LINE, MAX_LINES_PER_PAGE)
    events = []
    for lines in pages:
        inds = [j for ln in lines for j in ln]
        if not inds: continue
        if inds[-1] >= n or inds[0] >= n: continue
        t0_word = token_times[inds[0]][0]
        t1_word = token_times[inds[-1]][1]
        left_bound = t0_word
        right_bound = t1_word
        if inds[0] - 1 >= 0: left_bound = max(left_bound, mids[inds[0] - 1])
        if inds[-1] < n - 1: right_bound = min(right_bound, mids[inds[-1]])
        
        min_sec = 0.9 
        if right_bound - left_bound < min_sec: right_bound = left_bound + min_sec
        t0, t1 = left_bound, right_bound

        parts = []
        for ln in lines:
            clean_words = []
            for j in ln:
                clean_word = re.sub(r"[^a-zA-Zа-яА-Я0-9іїєґІЇЄҐ']+", "", tokens[j])
                if clean_word: clean_words.append(clean_word.upper())
            if clean_words: parts.append(escape_ass_text(" ".join(clean_words)))
        if not parts: continue
        txt = r"\N".join(parts)
        events.append((t0, t1, txt))
        
    if not events: return []
    fixed_events = []
    for i in range(len(events)):
        (t0, t1, txt) = events[i]
        next_t0 = float('inf')
        if i + 1 < len(events): next_t0 = events[i+1][0]
        if t1 > next_t0: t1 = next_t0
        if t1 <= t0: t1 = t0 + 0.1
        fixed_events.append((t0, t1, txt))
    return fixed_events

def write_ass_styled(out_path, events, style_settings):
    log.info(f"Генерація стилізованого ASS файлу...")
    
    # [!!! НОВЕ: ВІДСТУП !!!]
    y_offset_percent = style_settings.get('margin_bottom', 30) # За замовчуванням 30%
    target_w = 1080; target_h = 1920
    y_pos = int(target_h - (target_h * (y_offset_percent / 100.0)))
    x_pos = target_w // 2
    
    fontsize = style_settings.get('fontsize', 93)
    fontcolor = style_settings.get('fontcolor', '&H00FFFFFF')
    fontname = style_settings.get('fontname', 'Peace Sans')
    
    style_string = (
        f"Style: Default,{fontname},{fontsize},"
        f"{fontcolor},&H00FFFFFF,&H00000000,&H64000000,"
        "1,0,0,0,100,100,0,0,1,"
        "2,1,5,"
        "10,10,10,0"
    )

    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {target_w}", f"PlayResY: {target_h}",
           "ScaledBorderAndShadow: yes\n", "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
           "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
           "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           style_string + "\n", "[Events]",
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    if not events:
        ass.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0000,0000,0000,,{{\pos({x_pos},{y_pos})}}ПОМИЛКА: НЕ ЗНАЙДЕНО ТЕКСТУ")
    for (t0, t1, text) in events:
        ass_line = f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Default,,0000,0000,0000,,{{\pos({x_pos},{y_pos})}}{text}"
        ass.append(ass_line)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass))

def process_video_with_edits(video_path, all_words_original, edited_text, crf, style_settings):
    tokens, manual_starts = _parse_transcript_for_tokens(edited_text) 
    tokens_for_align = [re.sub(r"[^\w\s']+", "", t) for t in tokens]
    token_times = _align_tokens(all_words_original, tokens_for_align)
    events = _events_from_layout(tokens, token_times, manual_starts) 
    
    tmp_dir = tempfile.mkdtemp(prefix="sub_bot_")
    ass_path = os.path.join(tmp_dir, "subs.ass")
    write_ass_styled(ass_path, events, style_settings) 

    ff = find_ffmpeg()
    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(tmp_dir, f"{basename}_subs.mp4")
    sub_escaped = escape_for_subtitles_filter(ass_path)
    fontsdir_path = os.path.abspath("fonts")
    fontsdir_escaped = escape_for_subtitles_filter(fontsdir_path)
    
    # [!!! ВИПРАВЛЕННЯ ШРИФТІВ !!!]
    # Явно вказуємо папку шрифтів, якщо вона є
    if os.path.exists(fontsdir_path):
        vf_filter = f"subtitles='{sub_escaped}':fontsdir='{fontsdir_escaped}'"
    else:
        vf_filter = f"subtitles='{sub_escaped}'"

    cmd = [
        ff, "-y", "-i", video_path, "-vf", vf_filter,
        "-c:v", "libx264", 
        "-crf", "28", # Швидкість для Render
        "-preset", "ultrafast", 
        "-c:a", "copy",
        out_path
    ]
    
    log.info(f"Запуск FFmpeg: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    
    if proc.returncode != 0 or not os.path.isfile(out_path):
        log.error(f"Помилка FFmpeg:\n{proc.stdout[-1500:]}")
        raise RuntimeError(f"FFmpeg завершився з помилкою (код {proc.returncode}).")

    return out_path, tmp_dir

# --- Функції Telegram Бота ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привіт! Надішліть мені відео або аудіо для обробки.")
    context.user_data.clear()
    return ConversationHandler.END

async def handle_new_video_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обробка кнопки 'Завантажити нове відео'."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Добре! Надішліть нове відео або аудіо. 📤")
    context.user_data.clear()
    return ConversationHandler.END

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    video_input = message.video or message.document
    
    if not video_input:
        await message.reply_text("Будь ласка, надішліть відеофайл.")
        return STATE_RECEIVE_VIDEO if context.user_data else ConversationHandler.END

    try:
        file_name = video_input.file_name or "video.mp4"
        await message.reply_text("Завантажую відео... ⏳")

        new_file = await context.bot.get_file(video_input.file_id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_name}") as tmp_video_file:
            tmp_video_file_name = tmp_video_file.name
            await new_file.download_to_drive(tmp_video_file_name)
        
        # [!!! ЗМІНЕНО ТЕКСТ !!!]
        await message.reply_text("Витягую аудіо та розпізнаю текст... 🚀")
        
        ff = find_ffmpeg()
        audio_path = tmp_video_file_name + ".mp3"
        subprocess.run([ff, "-i", tmp_video_file_name, "-vn", "-acodec", "libmp3lame", "-q:a", "4", "-y", audio_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            original_text, all_words = transcribe_with_groq(audio_path)
        except Exception as e:
            log.error(f"Помилка Groq: {e}")
            await message.reply_text(f"Помилка API: {e}")
            return ConversationHandler.END
        finally:
            if os.path.exists(audio_path): os.remove(audio_path)

        if not all_words:
            await message.reply_text("Не вдалося розпізнати текст. 😢")
            return ConversationHandler.END

        clean_text = re.sub(r"[^\w\s']+", "", original_text).lower() 

        context.user_data['video_path'] = tmp_video_file_name
        context.user_data['all_words'] = all_words
        context.user_data['clean_text'] = clean_text 
        
        context.user_data['style_fontsize'] = 93
        context.user_data['style_color_name'] = 'Білий'
        context.user_data['style_color_value'] = '&H00FFFFFF'
        context.user_data['style_font_name'] = STYLE_FONTS[0]
        context.user_data['style_margin_bottom'] = 30 # За замовчуванням 30%

        # [!!! РОЗБИТТЯ ДОВГОГО ТЕКСТУ !!!]
        await message.reply_text(
            f"Ось розпізнаний текст (без пунктуації).\n\n"
            f"👉 Надішліть виправлений текст або 'ОК'.\n\n"
            f"Текст для копіювання нижче 👇",
            parse_mode='Markdown'
        )
        
        if len(clean_text) > 4000:
            for i in range(0, len(clean_text), 4000):
                await message.reply_text(clean_text[i:i+4000])
        else:
            await message.reply_text(f"{clean_text}")
        
        text_menu, keyboard = _get_style_menu(context.user_data)
        await message.reply_text(text_menu, reply_markup=keyboard, parse_mode='Markdown')

        return STATE_RECEIVE_EDIT

    except Exception as e:
        log.error(f"Помилка (handle_video): {e}", exc_info=True)
        await message.reply_text("Сталася помилка при обробці відео.")
        return ConversationHandler.END

async def handle_audio_transcription(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    audio_input = message.audio or (message.document if message.document and 'audio' in message.document.mime_type else None) or message.voice
    
    if not audio_input:
        await message.reply_text("Надішліть аудіофайл або голосове повідомлення.")
        return

    try:
        await message.reply_text("Завантажую аудіо... ⏳")
        new_file = await context.bot.get_file(audio_input.file_id)
        file_name = getattr(audio_input, 'file_name', 'voice.ogg') or 'audio.mp3'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_name}") as tmp_audio_file:
            tmp_path = tmp_audio_file.name
            await new_file.download_to_drive(tmp_path)
            
        # [!!! ЗМІНЕНО ТЕКСТ !!!]
        await message.reply_text("Розпізнаю текст... 🚀")

        original_text, _ = transcribe_with_groq(tmp_path) 
        clean_text = re.sub(r"[^\w\s']+", "", original_text).lower() 

        await message.reply_text(f"✅ **Готово!**\n\nТекст для копіювання 👇", parse_mode='Markdown')
        
        if len(clean_text) > 4000:
            for i in range(0, len(clean_text), 4000):
                await message.reply_text(clean_text[i:i+4000])
        else:
            await message.reply_text(f"{clean_text}")

        if os.path.exists(tmp_path): os.remove(tmp_path)

    except Exception as e:
        log.error(f"Помилка аудіо: {e}")
        await message.reply_text(f"Помилка: {e}")

async def handle_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_reply = update.message.text
    video_path = context.user_data.get('video_path')
    all_words = context.user_data.get('all_words')
    clean_text = context.user_data.get('clean_text')
    tmp_dir = None 
    keep_files_flag = False

    try:
        if not video_path or not all_words or not clean_text:
            await update.message.reply_text("Помилка стану. Будь ласка, надішліть відео знову.")
            return ConversationHandler.END

        if user_reply.lower() in ['окей', 'ок', 'ok']:
            text_to_process = clean_text
            await update.message.reply_text("Прийнято! Оброблюю... (Крок 2/3) ⚙️")
        else:
            text_to_process = user_reply
            await update.message.reply_text("Прийнято! Оброблюю ваш текст... (Крок 2/3) ⚙️")

        style_settings = {
            'fontsize': context.user_data.get('style_fontsize', 93),
            'fontcolor': context.user_data.get('style_color_value', '&H00FFFFFF'),
            'fontname': context.user_data.get('style_font_name', STYLE_FONTS[0]),
            'margin_bottom': context.user_data.get('style_margin_bottom', 30)
        }

        # Запускаємо обробку в окремому потоці
        loop = asyncio.get_running_loop()
        processed_path, tmp_dir = await loop.run_in_executor(
            None,
            process_video_with_edits, 
            video_path, 
            all_words, 
            text_to_process, 
            DEFAULT_CRF,
            style_settings 
        )
        
        context.user_data['tmp_dir'] = tmp_dir

        processed_file_size_mb = os.path.getsize(processed_path) / (1024 * 1024)
        
        if processed_file_size_mb > 49.0:
            log.warning(f"Файл занадто великий: {processed_file_size_mb:.2f} MB")
            await update.message.reply_text(
                f"✅ **Готово!**\n\n"
                f"Файл ({processed_file_size_mb:.2f} МБ) завеликий для Telegram.\n"
                f"Якщо ви на сервері - ви його не зможете забрати. Якщо локально - він тут:\n"
                f"`{processed_path}`",
                parse_mode='Markdown'
            )
            keep_files_flag = True
        else:
            # [!!! ПОВІДОМЛЕННЯ ПРО ЧАС !!!]
            await update.message.reply_text("Готово! Надсилаю відео... (Це може зайняти декілька хвилин) ⏳🚀")
            await context.bot.send_video(
                chat_id=update.message.chat_id,
                video=open(processed_path, 'rb'),
                filename=os.path.basename(processed_path),
                reply_to_message_id=update.message.message_id,
                read_timeout=120, 
                write_timeout=120, 
                connect_timeout=120
            )
            log.info(f"Відео надіслано: {update.message.chat_id}")

    except Exception as e:
        log.error(f"Помилка (handle_edit): {e}", exc_info=True)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Помилка: {e}")
    
    finally:
        video_path = context.user_data.get('video_path')
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        
        if not keep_files_flag:
            tmp_dir = context.user_data.get('tmp_dir')
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            
        context.user_data.clear()
        return ConversationHandler.END

# --- Стилі та Меню ---

STYLE_COLORS = {
    'Білий': '&H00FFFFFF',
    'Жовтий': '&H0000FFFF',
    'Червоний': '&H000000FF',
    'Зелений': '&H0000FF00',
    'Синій': '&H00FF0000',
}
COLOR_NAMES = list(STYLE_COLORS.keys())

STYLE_FONTS = [
    "Peace Sans",
    "Impact",
    "OpenSans-Light",
    "Franklin Gothic Heavy"
]

MARGIN_OPTIONS = [10, 15, 20, 25, 30, 40]

def _get_style_menu(user_data):
    size = user_data.get('style_fontsize', 93)
    color_name = user_data.get('style_color_name', 'Білий')
    font_name = user_data.get('style_font_name', 'Peace Sans')
    margin = user_data.get('style_margin_bottom', 30)
    
    text = (
        f"🎨 **Налаштування стилю**\n\n"
        f"Шрифт: `{font_name}`\n"
        f"Розмір: `{size}`\n"
        f"Колір: `{color_name}`\n"
        f"Відступ знизу: `{margin}%`\n\n"
        f"_(Налаштування застосуються після відправки тексту)_"
    )
    
    keyboard_size = [
        InlineKeyboardButton("Розмір -", callback_data='style_size_minus'),
        InlineKeyboardButton("Розмір +", callback_data='style_size_plus'),
    ]
    keyboard_color = [
        InlineKeyboardButton("‹ Колір", callback_data='style_color_prev'),
        InlineKeyboardButton("Колір ›", callback_data='style_color_next'),
    ]
    keyboard_font = [
        InlineKeyboardButton("‹ Шрифт", callback_data='style_font_prev'),
        InlineKeyboardButton("Шрифт ›", callback_data='style_font_next'),
    ]
    # [!!! НОВЕ: Кнопки для відступу !!!]
    keyboard_margin = [
        InlineKeyboardButton("‹ Відступ", callback_data='style_margin_prev'),
        InlineKeyboardButton("Відступ ›", callback_data='style_margin_next'),
    ]
    
    # [!!! НОВЕ: Кнопка скасування !!!]
    keyboard_cancel = [
        InlineKeyboardButton("❌ Завантажити нове відео", callback_data='new_video')
    ]
    
    return text, InlineKeyboardMarkup([keyboard_size, keyboard_color, keyboard_font, keyboard_margin, keyboard_cancel])

async def handle_style_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_data = context.user_data
    
    # [!!! Обробка кнопки "Нове відео" !!!]
    if data == 'new_video':
        await query.edit_message_text("Дію скасовано. Надішліть нове відео або аудіо. 📤")
        context.user_data.clear()
        return ConversationHandler.END
    
    if data == 'style_size_plus':
        user_data['style_fontsize'] = user_data.get('style_fontsize', 93) + 5
    elif data == 'style_size_minus':
        user_data['style_fontsize'] = max(10, user_data.get('style_fontsize', 93) - 5)
        
    elif data in ['style_color_prev', 'style_color_next']:
        curr = user_data.get('style_color_name', 'Білий')
        try: idx = COLOR_NAMES.index(curr)
        except: idx = 0
        if data == 'style_color_next': idx = (idx + 1) % len(COLOR_NAMES)
        else: idx = (idx - 1) % len(COLOR_NAMES)
        new_name = COLOR_NAMES[idx]
        user_data['style_color_name'] = new_name
        user_data['style_color_value'] = STYLE_COLORS[new_name]

    elif data in ['style_font_prev', 'style_font_next']:
        curr = user_data.get('style_font_name', 'Peace Sans')
        try: idx = STYLE_FONTS.index(curr)
        except: idx = 0
        if data == 'style_font_next': idx = (idx + 1) % len(STYLE_FONTS)
        else: idx = (idx - 1) % len(STYLE_FONTS)
        user_data['style_font_name'] = STYLE_FONTS[idx]
        
    # [!!! Логіка відступу !!!]
    elif data in ['style_margin_prev', 'style_margin_next']:
        curr = user_data.get('style_margin_bottom', 30)
        try: idx = MARGIN_OPTIONS.index(curr)
        except: idx = 4 # 30%
        if data == 'style_margin_next': idx = (idx + 1) % len(MARGIN_OPTIONS)
        else: idx = (idx - 1) % len(MARGIN_OPTIONS)
        user_data['style_margin_bottom'] = MARGIN_OPTIONS[idx]

    text, keyboard = _get_style_menu(user_data)
    try:
        await query.edit_message_text(text, reply_markup=keyboard, parse_mode='Markdown')
    except Exception:
        pass

# --- WEB SERVER ---
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Bot is alive!')

def start_web_server():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
    log.info(f"Web server started on port {port}")
    server.serve_forever()

if __name__ == "__main__":
    if "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" in TELEGRAM_BOT_TOKEN:
        log.error("Вкажіть TELEGRAM_BOT_TOKEN!")
        sys.exit(1)
        
    log.info("Створюємо додаток бота...")
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio_transcription))

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start_command), 
            MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video) 
        ],
        states={
            STATE_RECEIVE_EDIT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit),
                # Додаємо обробку new_video сюди
                CallbackQueryHandler(handle_style_buttons, pattern='^(style_|new_video)')
            ],
        },
        fallbacks=[
            CommandHandler('start', start_command), 
            CommandHandler('cancel', cancel_command) 
        ],
    )
    application.add_handler(conv_handler)

    threading.Thread(target=start_web_server, daemon=True).start()
    log.info("Бот запускається...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
