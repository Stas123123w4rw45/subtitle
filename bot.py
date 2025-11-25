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
import json
import gc
from groq import Groq 

SETTINGS_FILE = "user_settings.json"

def load_settings(chat_id):
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
                return data.get(str(chat_id), {})
    except Exception as e:
        log.error(f"Error loading settings: {e}")
    return {}

def save_settings(chat_id, settings):
    try:
        data = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                try: data = json.load(f)
                except: pass
        
        data[str(chat_id)] = settings
        
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        log.error(f"Error saving settings: {e}") 

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
DEFAULT_CRF = "20" 
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

def generate_styled_events(tokens, token_times, style_settings):
    """
    Generates ASS events based on layout settings (WPL, Max Lines) and applies
    effects (Karaoke, Animation) if enabled.
    """
    wpl = style_settings.get('wpl', WORDS_PER_LINE)
    max_lines = style_settings.get('max_lines', MAX_LINES_PER_PAGE)
    karaoke = style_settings.get('karaoke', False)
    animation = style_settings.get('animation', False)
    # highlight_color is applied via \c tag. Default is taken from settings or hardcoded if missing.
    highlight_color = style_settings.get('highlight_color', '&H0000FFFF') # Yellow default for karaoke

    n = len(tokens)
    if n == 0 or not token_times: return []

    # 1. Calculate Layout (Pages/Lines)
    # We use a simplified manual_starts logic (assuming none for now or passed if needed)
    manual_starts = set() 
    pages = _pages_from_manual_or_auto(n, manual_starts, wpl, max_lines)

    events = []
    
    for page_lines in pages:
        if not page_lines: continue
        
        # Flatten lines to get start/end indices for the whole page/event
        all_indices = [idx for line in page_lines for idx in line]
        if not all_indices: continue
        
        start_idx = all_indices[0]
        end_idx = all_indices[-1]
        
        if start_idx >= len(token_times) or end_idx >= len(token_times): continue

        t_start = token_times[start_idx][0]
        t_end = token_times[end_idx][1]
        
        # Pad time slightly
        t_start = max(0, t_start - 0.1)
        t_end = t_end + 0.1
        duration_ms = (t_end - t_start) * 1000

        # Build Text
        lines_text = []
        for line_indices in page_lines:
            line_parts = []
            for idx in line_indices:
                if idx >= len(tokens): continue
                
                token = tokens[idx]
                clean_token = re.sub(r"[^a-zA-Zа-яА-Я0-9іїєґІЇЄҐ']+", "", token)
                if not clean_token: continue
                
                # Word Timing
                w_start = token_times[idx][0]
                w_end = token_times[idx][1]
                
                # Relative times in ms for tags
                rel_start = int((w_start - t_start) * 1000)
                rel_end = int((w_end - t_start) * 1000)
                
                prefix = ""
                suffix = ""
                
                # Apply Effects
                if karaoke:
                    # \t(t1,t2,tags) - animate color change
                    # We want it to be Highlighted during [rel_start, rel_end]
                    # And Normal before/after.
                    # Since \t is cumulative, it's tricky. 
                    # Simpler approach: \1c&H...& for highlight, then revert.
                    # But \t is better for smooth or precise timing.
                    # Let's use a simple hard switch:
                    # {\t(start,start,\1c&HHighlight&)}{\t(end,end,\1c&HNormal&)}
                    
                    # Note: ASS colors are BGR. 
                    normal_color = style_settings.get('fontcolor', '&H00FFFFFF')
                    
                    prefix += f"{{\\t({rel_start},{rel_start},\\1c{highlight_color})}}{{\\t({rel_end},{rel_end},\\1c{normal_color})}}"

                if animation:
                    # Pop up effect: Scale up to 120% then back to 100%
                    # {\t(start, mid, \fscx120\fscy120)}{\t(mid, end, \fscx100\fscy100)}
                    mid = (rel_start + rel_end) // 2
                    prefix += f"{{\\t({rel_start},{mid},\\fscx120\\fscy120)}}{{\\t({mid},{rel_end},\\fscx100\\fscy100)}}"

                line_parts.append(f"{prefix}{clean_token.upper()}{suffix}")
            
            if line_parts:
                lines_text.append(" ".join(line_parts))
        
        if lines_text:
            full_text = r"\N".join(lines_text)
            events.append((t_start, t_end, full_text))

    return events

def write_ass_styled(out_path, events, style_settings):
    log.info(f"Генерація стилізованого ASS файлу...")
    
    y_offset_percent = style_settings.get('margin_bottom', 30)
    target_w = 1080; target_h = 1920
    y_pos = int(target_h - (target_h * (y_offset_percent / 100.0)))
    x_pos = target_w // 2
    
    fontsize = style_settings.get('fontsize', 93)
    fontcolor = style_settings.get('fontcolor', '&H00FFFFFF')
    fontname = style_settings.get('fontname', 'Peace Sans')
    
    # Shadow and Outline settings
    shadow_size = 0 if not style_settings.get('shadow_enabled', True) else 4
    outline_size = 0 if not style_settings.get('outline_enabled', True) else 3
    
    # Main Style
    style_string = (
        f"Style: Default,{fontname},{fontsize},"
        f"{fontcolor},&H00FFFFFF,&H00000000,&H64000000,"
        f"1,0,0,0,100,100,0,0,1,"
        f"{outline_size},{shadow_size},5," # Outline, Shadow, Alignment (5=Center)
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
        ass.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0000,0000,0000,,{{\\pos({x_pos},{y_pos})}}ПОМИЛКА: НЕ ЗНАЙДЕНО ТЕКСТУ")
    
    for (t0, t1, text) in events:
        ass_line = f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Default,,0000,0000,0000,,{{\\pos({x_pos},{y_pos})}}{text}"
        ass.append(ass_line)
        
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass))

def process_video_with_edits(video_path, all_words_original, edited_text, crf, style_settings):
    tokens, manual_starts = _parse_transcript_for_tokens(edited_text) 
    tokens_for_align = [re.sub(r"[^\w\s']+", "", t) for t in tokens]
    token_times = _align_tokens(all_words_original, tokens_for_align)
    # Generate events using new logic
    events = generate_styled_events(tokens, token_times, style_settings)
    
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
        ff, "-y", "-i", video_path, "-vf", f"scale='min(1080,iw)':-2,{vf_filter}",
        "-c:v", "libx264", 
        "-crf", "23", # Трохи збільшуємо CRF для меншого навантаження (було 20)
        "-preset", "superfast", # superfast менше їсть пам'яті ніж ultrafast іноді, або так само
        "-threads", "2", # ОБМЕЖЕННЯ ПОТОКІВ для економії RAM
        "-max_muxing_queue_size", "1024",
        "-movflags", "+faststart",
        "-c:a", "copy",
        out_path
    ]
    
    log.info(f"Запуск FFmpeg: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    
    if proc.returncode != 0 or not os.path.isfile(out_path):
        log.error(f"Помилка FFmpeg:\n{proc.stdout[-1500:]}")
        raise RuntimeError(f"FFmpeg завершився з помилкою (код {proc.returncode}).")

    return out_path, tmp_dir

def get_video_duration(file_path):
    """Returns video duration in seconds using ffprobe."""
    try:
        # Try using ffprobe if available, otherwise estimate or fail
        # We can use ffmpeg to get duration from stderr
        ff = find_ffmpeg()
        cmd = [ff, "-i", file_path]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, encoding='utf-8')
        # Search for "Duration: 00:00:05.00"
        match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", result.stderr)
        if match:
            h, m, s = map(float, match.groups())
            return h * 3600 + m * 60 + s
    except Exception as e:
        log.error(f"Error getting duration: {e}")
    return 0

def compress_video(input_path, target_size_mb=49.0):
    """Compresses video to target size using bitrate control."""
    log.info(f"Compressing {input_path} to {target_size_mb}MB...")
    
    duration = get_video_duration(input_path)
    if duration <= 0:
        log.error("Could not determine duration for compression.")
        return input_path # Return original if fail

    # Calculate target bitrate
    # Size = (VideoBitrate + AudioBitrate) * Duration / 8
    # TargetBits = TargetMB * 8 * 1024 * 1024
    # TargetBitrate = TargetBits / Duration
    
    target_total_bitrate_kbit = (target_size_mb * 8 * 1024) / duration
    audio_bitrate_kbit = 128
    video_bitrate_kbit = target_total_bitrate_kbit - audio_bitrate_kbit
    
    if video_bitrate_kbit < 100: video_bitrate_kbit = 100 # Minimum floor
    
    log.info(f"Duration: {duration}s, Target Bitrate: {video_bitrate_kbit:.0f}k")
    
    ff = find_ffmpeg()
    dir_name = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(dir_name, f"{base_name}_compressed.mp4")
    
    # Single pass with bitrate control is usually enough for this purpose
    # We use -maxrate and -bufsize to constrain it
    cmd = [
        ff, "-y", "-i", input_path,
        "-c:v", "libx264",
        "-b:v", f"{video_bitrate_kbit}k",
        "-maxrate", f"{video_bitrate_kbit * 1.5}k",
        "-bufsize", f"{video_bitrate_kbit * 2}k",
        "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    return input_path

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
        
        # Load saved settings
        saved_settings = load_settings(message.chat_id)
        
        context.user_data['style_fontsize'] = saved_settings.get('fontsize', 93)
        context.user_data['style_color_name'] = saved_settings.get('color_name', 'Білий')
        context.user_data['style_color_value'] = saved_settings.get('color_value', '&H00FFFFFF')
        context.user_data['style_font_name'] = saved_settings.get('font_name', STYLE_FONTS[0])
        context.user_data['style_margin_bottom'] = saved_settings.get('margin_bottom', 30)
        context.user_data['style_shadow_enabled'] = saved_settings.get('shadow_enabled', True)
        context.user_data['style_outline_enabled'] = saved_settings.get('outline_enabled', True)
        context.user_data['style_wpl'] = saved_settings.get('wpl', WORDS_PER_LINE)
        context.user_data['style_max_lines'] = saved_settings.get('max_lines', MAX_LINES_PER_PAGE)
        context.user_data['style_animation'] = saved_settings.get('animation', False)
        context.user_data['style_karaoke'] = saved_settings.get('karaoke', False)

        # [!!! РОЗБИТТЯ ДОВГОГО ТЕКСТУ !!!]
        await message.reply_text(
            f"Ось розпізнаний текст (без пунктуації).\n\n"
            f"👉 Надішліть виправлений текст (подвійний пробіл - новий рядок) або 'ОК'.\n\n"
            f"Текст для копіювання нижче 👇",
            parse_mode='Markdown'
        )
        
        if len(clean_text) > 4000:
            for i in range(0, len(clean_text), 4000):
                await message.reply_text(clean_text[i:i+4000])
        else:
            await message.reply_text(f"{clean_text}")
        
        text_menu, keyboard = _get_settings_menu(context.user_data, 'main')
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
    clean_text = context.user_data.get('clean_text')

    if user_reply.lower() in ['окей', 'ок', 'ok']:
        text_to_process = clean_text
    else:
        text_to_process = user_reply
        
    context.user_data['text_to_process'] = text_to_process
    
    # Show menu again with updated text confirmation
    await update.message.reply_text("Текст прийнято! Налаштуйте стиль і натисніть 'Готово'. 👇")
    text_menu, keyboard = _get_settings_menu(context.user_data, 'main')
    await update.message.reply_text(text_menu, reply_markup=keyboard, parse_mode='Markdown')
    return STATE_RECEIVE_EDIT

async def run_processing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Executed when user clicks 'Done'.
    """
    video_path = context.user_data.get('video_path')
    all_words = context.user_data.get('all_words')
    text_to_process = context.user_data.get('text_to_process', context.user_data.get('clean_text'))
    
    tmp_dir = None 
    keep_files_flag = False
    
    # Determine chat_id/message_id for reply
    if update.message:
        chat_id = update.message.chat_id
        reply_to = update.message.message_id
    else:
        chat_id = update.effective_chat.id
        reply_to = None # Callback query doesn't have a message ID to reply to in the same way

    try:
        if not video_path or not all_words:
            await context.bot.send_message(chat_id, "Помилка стану. Надішліть відео знову.")
            return

        style_settings = {
            'fontsize': context.user_data.get('style_fontsize', 93),
            'fontcolor': context.user_data.get('style_color_value', '&H00FFFFFF'),
            'fontname': context.user_data.get('style_font_name', STYLE_FONTS[0]),
            'margin_bottom': context.user_data.get('style_margin_bottom', 30),
            'shadow_enabled': context.user_data.get('style_shadow_enabled', True),
            'outline_enabled': context.user_data.get('style_outline_enabled', True),
            'wpl': context.user_data.get('style_wpl', 2),
            'max_lines': context.user_data.get('style_max_lines', 1),
            'animation': context.user_data.get('style_animation', False),
            'karaoke': context.user_data.get('style_karaoke', False)
        }

        # Запускаємо обробку в окремому потоці
        loop = asyncio.get_running_loop()
        
        # Force GC before heavy operation
        gc.collect()
        
        processed_path, tmp_dir = await loop.run_in_executor(
            None,
            process_video_with_edits, 
            video_path, 
            all_words, 
            text_to_process, 
            DEFAULT_CRF,
            style_settings 
        )
        
        # Force GC after heavy operation
        gc.collect()
        
        context.user_data['tmp_dir'] = tmp_dir

        processed_file_size_mb = os.path.getsize(processed_path) / (1024 * 1024)
        
        if processed_file_size_mb > 49.0:
            log.warning(f"Файл занадто великий: {processed_file_size_mb:.2f} MB. Починаю стиснення...")
            await context.bot.send_message(chat_id, "Файл великий (>50MB). Стискаю, щоб надіслати... 📉")
            
            # Run compression
            compressed_path = await loop.run_in_executor(
                None, 
                compress_video, 
                processed_path, 
                48.0 # Target slightly less than 49 to be safe
            )
            
            # Check new size
            new_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
            if new_size_mb > 49.5:
                 log.warning(f"Стиснення не допомогло: {new_size_mb:.2f} MB")
                 await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ **Не вдалося стиснути достатньо.**\nРозмір: {new_size_mb:.2f} МБ.\n`{compressed_path}`",
                    parse_mode='Markdown'
                )
                 keep_files_flag = True
            else:
                log.info(f"Стиснуто до {new_size_mb:.2f} MB")
                await context.bot.send_video(
                    chat_id=chat_id,
                    video=open(compressed_path, 'rb'),
                    filename=os.path.basename(compressed_path),
                    read_timeout=120, 
                    write_timeout=120, 
                    connect_timeout=120
                )
        else:
            await context.bot.send_message(chat_id, "Готово! Надсилаю відео... ⏳🚀")
            await context.bot.send_video(
                chat_id=chat_id,
                video=open(processed_path, 'rb'),
                filename=os.path.basename(processed_path),
                read_timeout=120, 
                write_timeout=120, 
                connect_timeout=120
            )
            log.info(f"Відео надіслано: {chat_id}")

    except Exception as e:
        log.error(f"Помилка (run_processing): {e}", exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=f"Помилка: {e}")
    
    finally:
        video_path = context.user_data.get('video_path')
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        
        if not keep_files_flag:
            tmp_dir = context.user_data.get('tmp_dir')
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            
        context.user_data.clear()

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

# --- UI & MENUS ---

def _get_settings_menu(user_data, menu_state='main'):
    """
    Generates text and keyboard for different menu states:
    'main', 'style', 'layout', 'effects'
    """
    # Load current values
    fontsize = user_data.get('style_fontsize', 93)
    color_name = user_data.get('style_color_name', 'Білий')
    font_name = user_data.get('style_font_name', 'Peace Sans')
    margin = user_data.get('style_margin_bottom', 30)
    
    shadow = "✅" if user_data.get('style_shadow_enabled', True) else "❌"
    outline = "✅" if user_data.get('style_outline_enabled', True) else "❌"
    
    wpl = user_data.get('style_wpl', WORDS_PER_LINE)
    max_lines = user_data.get('style_max_lines', MAX_LINES_PER_PAGE)
    
    anim = "✅" if user_data.get('style_animation', False) else "❌"
    karaoke = "✅" if user_data.get('style_karaoke', False) else "❌"
    
    text = ""
    keyboard = []

    if menu_state == 'main':
        text = (
            f"⚙️ **Налаштування субтитрів**\n\n"
            f"🎨 Стиль: {font_name}, {fontsize}, {color_name}\n"
            f"📐 Макет: {wpl} слів/ряд, {max_lines} ряд/стор\n"
            f"✨ Ефекти: Анім {anim}, Караоке {karaoke}\n"
        )
        keyboard = [
            [InlineKeyboardButton("🎨 Стиль", callback_data='menu_style'),
             InlineKeyboardButton("📐 Макет", callback_data='menu_layout')],
            [InlineKeyboardButton("✨ Ефекти", callback_data='menu_effects')],
            [InlineKeyboardButton("✅ Готово (Обробити)", callback_data='process_done')],
            [InlineKeyboardButton("❌ Скасувати", callback_data='new_video')]
        ]

    elif menu_state == 'style':
        text = (
            f"🎨 **Стиль**\n\n"
            f"Шрифт: {font_name}\n"
            f"Розмір: {fontsize}\n"
            f"Колір: {color_name}\n"
            f"Тінь: {shadow} | Обводка: {outline}"
        )
        keyboard = [
            [InlineKeyboardButton("Шрифт ›", callback_data='set_font_next')],
            [InlineKeyboardButton("- Розмір", callback_data='set_size_minus'),
             InlineKeyboardButton("+ Розмір", callback_data='set_size_plus')],
            [InlineKeyboardButton("‹ Колір", callback_data='set_color_prev'),
             InlineKeyboardButton("Колір ›", callback_data='set_color_next')],
            [InlineKeyboardButton(f"Тінь {shadow}", callback_data='toggle_shadow'),
             InlineKeyboardButton(f"Обводка {outline}", callback_data='toggle_outline')],
            [InlineKeyboardButton("🔙 Назад", callback_data='menu_main')]
        ]

    elif menu_state == 'layout':
        text = (
            f"📐 **Макет**\n\n"
            f"Слів у рядку: {wpl}\n"
            f"Рядків на сторінці: {max_lines}\n"
            f"Відступ знизу: {margin}%"
        )
        keyboard = [
            [InlineKeyboardButton("- Слова", callback_data='set_wpl_minus'),
             InlineKeyboardButton("+ Слова", callback_data='set_wpl_plus')],
            [InlineKeyboardButton("- Рядки", callback_data='set_lines_minus'),
             InlineKeyboardButton("+ Рядки", callback_data='set_lines_plus')],
            [InlineKeyboardButton("- Відступ", callback_data='set_margin_minus'),
             InlineKeyboardButton("+ Відступ", callback_data='set_margin_plus')],
            [InlineKeyboardButton("🔙 Назад", callback_data='menu_main')]
        ]

    elif menu_state == 'effects':
        text = (
            f"✨ **Ефекти**\n\n"
            f"Pop-up Анімація: {anim}\n"
            f"Караоке (підсвітка): {karaoke}\n"
            f"_(Караоке змінює колір активного слова)_"
        )
        keyboard = [
            [InlineKeyboardButton(f"Анімація {anim}", callback_data='toggle_anim')],
            [InlineKeyboardButton(f"Караоке {karaoke}", callback_data='toggle_karaoke')],
            [InlineKeyboardButton("🔙 Назад", callback_data='menu_main')]
        ]

    return text, InlineKeyboardMarkup(keyboard)

async def handle_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_data = context.user_data
    
    # Navigation
    if data.startswith('menu_'):
        target = data.split('_')[1]
        text, markup = _get_settings_menu(user_data, target)
        await query.edit_message_text(text, reply_markup=markup, parse_mode='Markdown')
        return

    # Actions
    if data == 'process_done':
        # Trigger processing (simulate sending "OK")
        # We can't easily trigger the message handler, so we call the logic directly or ask user to confirm.
        # Better: Just say "Settings saved. Send 'OK' to process." or edit text to say "Processing..."
        # Actually, the user flow is: Send Video -> Send Text -> Menu -> (User clicks Done) -> Processing.
        # So we need to call the processing logic here.
        await query.edit_message_text("Налаштування збережено! Починаю обробку... ⚙️")
        
        # We need to trigger handle_edit logic. 
        # Since handle_edit expects a Message update, we can't call it directly with CallbackQuery.
        # We will extract the logic to a helper or just run it here.
        # Let's call a helper function `run_processing(update, context)`
        
        # [!!! FIX: RUN IN BACKGROUND !!!]
        # Use create_task to prevent blocking the webhook response
        asyncio.create_task(run_processing(update, context))
        return ConversationHandler.END

    if data == 'new_video':
        await query.edit_message_text("Дію скасовано. Надішліть нове відео. 📤")
        context.user_data.clear()
        return ConversationHandler.END

    # Style Setters
    if data == 'set_size_plus':
        user_data['style_fontsize'] = user_data.get('style_fontsize', 93) + 5
    elif data == 'set_size_minus':
        user_data['style_fontsize'] = max(10, user_data.get('style_fontsize', 93) - 5)
    
    elif data == 'set_font_next':
        curr = user_data.get('style_font_name', STYLE_FONTS[0])
        try: idx = STYLE_FONTS.index(curr)
        except: idx = 0
        idx = (idx + 1) % len(STYLE_FONTS)
        user_data['style_font_name'] = STYLE_FONTS[idx]

    elif data == 'set_color_next':
        curr = user_data.get('style_color_name', 'Білий')
        try: idx = COLOR_NAMES.index(curr)
        except: idx = 0
        idx = (idx + 1) % len(COLOR_NAMES)
        new_name = COLOR_NAMES[idx]
        user_data['style_color_name'] = new_name
        user_data['style_color_value'] = STYLE_COLORS[new_name]
    elif data == 'set_color_prev':
        curr = user_data.get('style_color_name', 'Білий')
        try: idx = COLOR_NAMES.index(curr)
        except: idx = 0
        idx = (idx - 1) % len(COLOR_NAMES)
        new_name = COLOR_NAMES[idx]
        user_data['style_color_name'] = new_name
        user_data['style_color_value'] = STYLE_COLORS[new_name]

    elif data == 'toggle_shadow':
        user_data['style_shadow_enabled'] = not user_data.get('style_shadow_enabled', True)
    elif data == 'toggle_outline':
        user_data['style_outline_enabled'] = not user_data.get('style_outline_enabled', True)

    # Layout Setters
    elif data == 'set_wpl_plus':
        user_data['style_wpl'] = user_data.get('style_wpl', 2) + 1
    elif data == 'set_wpl_minus':
        user_data['style_wpl'] = max(1, user_data.get('style_wpl', 2) - 1)
    
    elif data == 'set_lines_plus':
        user_data['style_max_lines'] = user_data.get('style_max_lines', 1) + 1
    elif data == 'set_lines_minus':
        user_data['style_max_lines'] = max(1, user_data.get('style_max_lines', 1) - 1)

    elif data == 'set_margin_plus':
        user_data['style_margin_bottom'] = min(50, user_data.get('style_margin_bottom', 30) + 5)
    elif data == 'set_margin_minus':
        user_data['style_margin_bottom'] = max(0, user_data.get('style_margin_bottom', 30) - 5)

    # Effects Setters
    elif data == 'toggle_anim':
        user_data['style_animation'] = not user_data.get('style_animation', False)
    elif data == 'toggle_karaoke':
        user_data['style_karaoke'] = not user_data.get('style_karaoke', False)

    # Save settings
    current_settings = {
        'fontsize': user_data.get('style_fontsize'),
        'color_name': user_data.get('style_color_name'),
        'color_value': user_data.get('style_color_value'),
        'font_name': user_data.get('style_font_name'),
        'margin_bottom': user_data.get('style_margin_bottom'),
        'shadow_enabled': user_data.get('style_shadow_enabled'),
        'outline_enabled': user_data.get('style_outline_enabled'),
        'wpl': user_data.get('style_wpl'),
        'max_lines': user_data.get('style_max_lines'),
        'animation': user_data.get('style_animation'),
        'karaoke': user_data.get('style_karaoke')
    }
    save_settings(query.message.chat_id, current_settings)

    # Refresh current menu
    # Need to know which menu we are in. 
    # A simple hack is to check the button that was clicked or store state.
    # But since we don't store menu state in user_data, we can infer or just default to 'main' 
    # or try to guess. 
    # Better: Pass the menu state in callback data? e.g. 'set_size_plus:style'
    # For now, let's infer based on the command group.
    
    next_menu = 'main'
    if data.startswith('set_size') or data.startswith('set_font') or data.startswith('set_color') or 'shadow' in data or 'outline' in data:
        next_menu = 'style'
    elif 'wpl' in data or 'lines' in data or 'margin' in data:
        next_menu = 'layout'
    elif 'anim' in data or 'karaoke' in data:
        next_menu = 'effects'
        
    text, markup = _get_settings_menu(user_data, next_menu)
    try:
        await query.edit_message_text(text, reply_markup=markup, parse_mode='Markdown')
    except Exception:
        pass



async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Скасовує поточну операцію та чистить файли."""
    video_path = context.user_data.get('video_path')
    tmp_dir = context.user_data.get('tmp_dir')
    
    if video_path and os.path.exists(video_path):
        try: os.remove(video_path)
        except: pass
        
    if tmp_dir and os.path.exists(tmp_dir):
        try: shutil.rmtree(tmp_dir)
        except: pass
        
    context.user_data.clear()
    await update.message.reply_text("Дію скасовано. ✅")
    return ConversationHandler.END

async def wait_for_network():
    """Waits for Telegram API to be reachable."""
    url = "https://api.telegram.org"
    retries = 0
    max_retries = 10
    
    log.info("Checking network connectivity...")
    import httpx
    
    while retries < max_retries:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.get(url)
                log.info("Network is UP! 🚀")
                return
        except Exception as e:
            retries += 1
            log.warning(f"Network check failed ({retries}/{max_retries}): {e}")
            await asyncio.sleep(2)
            
    log.error("Network check failed after max retries. Proceeding anyway...")

if __name__ == "__main__":
    if "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" in TELEGRAM_BOT_TOKEN:
        log.error("Вкажіть TELEGRAM_BOT_TOKEN!")
        sys.exit(1)
        
    log.info("Створюємо додаток бота...")
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).connect_timeout(30).read_timeout(30).build()
    
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
                CallbackQueryHandler(handle_settings_callback, pattern='^(menu_|set_|toggle_|process_|new_video)')
            ],
        },
        fallbacks=[
            CommandHandler('start', start_command), 
            CommandHandler('cancel', cancel_command) 
        ],
    )
    application.add_handler(conv_handler)

    # Check for Render environment
    RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")
    PORT = int(os.getenv("PORT", "7860"))

    if RENDER_EXTERNAL_URL:
        # Wait for network before starting webhook
        loop = asyncio.new_event_loop()
        loop.run_until_complete(wait_for_network())
        
        webhook_url = f"{RENDER_EXTERNAL_URL}/{TELEGRAM_BOT_TOKEN}"
        log.info(f"Starting in WEBHOOK mode. Port: {PORT}, URL: {RENDER_EXTERNAL_URL}")
        log.info(f"Setting webhook to: {webhook_url}")
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_BOT_TOKEN,
            webhook_url=webhook_url,
            allowed_updates=Update.ALL_TYPES
        )
    else:
        log.info("Starting in POLLING mode...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
