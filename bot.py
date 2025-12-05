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

# Settings file path - use Railway Volume if available, fallback to local
# On Railway with Volume mounted at /app/data, this will persist across restarts
SETTINGS_DIR = "/app/data" if os.path.exists("/app/data") else "."
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "user_settings.json")

# Ensure settings directory exists
try:
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    # Test write permissions
    test_file = os.path.join(SETTINGS_DIR, ".test_write")
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    print(f"✅ Settings directory ready: {SETTINGS_DIR}")
except Exception as e:
    print(f"⚠️ Settings directory issue: {e}, using current directory")
    SETTINGS_FILE = "user_settings.json"

def load_settings(chat_id):
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
                return data.get(str(chat_id), {})
    except Exception as e:
        print(f"Error loading settings: {e}")  # Use print before logging is configured
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
            json.dump(data, f, indent=2)  # Added indent for readability
    except Exception as e:
        print(f"Error saving settings: {e}")  

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
    # 1. Check local folder (priority for portable setup)
    local_ffmpeg = os.path.join(os.getcwd(), "ffmpeg")
    if os.path.isfile(local_ffmpeg) and os.access(local_ffmpeg, os.X_OK):
        log.info(f"Using local ffmpeg: {local_ffmpeg}")
        return local_ffmpeg
        
    # 2. Check PATH
    if shutil.which("ffmpeg"):
        log.info("Using ffmpeg from PATH")
        return "ffmpeg"
        
    log.error("FFmpeg not found!")
    return "ffmpeg" # Fallback

def escape_for_subtitles_filter(path):
    p = os.path.abspath(path)
    if os.name == 'nt':
        p = p.replace("\\", "/").replace(":", "\\:")
    return p

def create_rounded_rect_path(w, h, r):
    """Generates ASS vector path for a rounded rectangle centered at 0,0."""
    hw = w / 2
    hh = h / 2
    r = min(r, hw, hh)
    
    # Start top-left (after radius)
    path = f"m {-hw + r} {-hh} l {hw - r} {-hh} "
    # Top-right corner
    path += f"b {hw} {-hh} {hw} {-hh} {hw} {-hh + r} "
    # Right edge
    path += f"l {hw} {hh - r} "
    # Bottom-right corner
    path += f"b {hw} {hh} {hw} {hh} {hw - r} {hh} "
    # Bottom edge
    path += f"l {-hw + r} {hh} "
    # Bottom-left corner
    path += f"b {-hw} {hh} {-hw} {hh} {-hw} {hh - r} "
    # Left edge
    path += f"l {-hw} {-hh + r} "
    # Top-left corner
    path += f"b {-hw} {-hh} {-hw} {-hh} {-hw + r} {-hh} "
    
    return path

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

def _smart_layout(tokens, wpl=2, max_lines=1, fontsize=93):
    """
    Smart layout that respects:
    1. Punctuation (force page break after . ! ?)
    2. Max Width (18% margin -> ~700px safe zone)
    3. WPL (soft limit)
    4. Max Lines (hard limit)
    """
    SAFE_WIDTH = 1080 * 0.64  # 1080 - 18%*2 margins ~= 690px
    AVG_CHAR_WIDTH = fontsize * 0.55 # Heuristic
    
    pages = []
    current_page = []
    current_line = []
    current_line_width = 0
    
    for i, token in enumerate(tokens):
        # Clean token for width calc
        clean_t = re.sub(r"[^a-zA-Zа-яА-Я0-9іїєґІЇЄҐ']+", "", token)
        word_width = len(clean_t) * AVG_CHAR_WIDTH
        
        # Check if we need to wrap to new line
        # 1. WPL limit reached
        # 2. Width limit reached
        force_new_line = False
        if len(current_line) >= wpl:
            force_new_line = True
        elif current_line_width + word_width > SAFE_WIDTH:
            force_new_line = True
            
        if force_new_line:
            if current_line:
                current_page.append(current_line)
            current_line = []
            current_line_width = 0
            
            # Check if page is full
            if len(current_page) >= max_lines:
                pages.append(current_page)
                current_page = []
        
        # Add to current line
        current_line.append(i)
        current_line_width += word_width + (fontsize * 0.2) # Add space width
        
        # Check for sentence ending punctuation
        # We look at the raw token (it might have punctuation)
        if token and token[-1] in ".!?,:—-":
            # Finish line
            current_page.append(current_line)
            current_line = []
            current_line_width = 0
            # Finish page
            pages.append(current_page)
            current_page = []
            
    # Flush remaining
    if current_line:
        current_page.append(current_line)
    if current_page:
        pages.append(current_page)
        
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
    highlight_color = style_settings.get('highlight_color_value', '&H0000FFFF') # Yellow default for karaoke
    
    # Box Settings
    box_enabled = style_settings.get('box_enabled', False)
    box_color = str(style_settings.get('box_color_value', '&H00FFFFFF'))
    box_opacity_hex = str(style_settings.get('box_opacity', '&H00'))

    n = len(tokens)
    if n == 0 or not token_times: return []

    # 1. Calculate Layout (Pages/Lines)
    # Use smart layout
    fontsize = style_settings.get('fontsize', 93)
    pages = _smart_layout(tokens, wpl, max_lines, fontsize)

    events = []
    
    # For box positioning
    target_w = 1080
    
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

        # Build Text & Boxes
        lines_text = []
        page_boxes = []
        
        # Calculate total width of each line to center it
        # This is an approximation for box positioning
        
        for line_idx, line_indices in enumerate(page_lines):
            line_parts = []
            
            # Calculate total line width for centering
            total_line_width = 0
            word_widths = []
            space_width = fontsize * 0.2
            
            for idx in line_indices:
                if idx >= len(tokens): 
                    word_widths.append(0)
                    continue
                token = tokens[idx]
                clean_token = re.sub(r"[^a-zA-Zа-яА-Я0-9іїєґІЇЄҐ']+", "", token)
                # Heuristic width
                ww = len(clean_token) * (fontsize * 0.55) # Same heuristic as smart layout
                word_widths.append(ww)
                total_line_width += ww + space_width
            
            if total_line_width > 0: total_line_width -= space_width
            
            # Start X (Centered)
            current_x = (target_w - total_line_width) / 2
            
            prev_box_x = None
            
            for i, idx in enumerate(line_indices):
                if idx >= len(tokens): continue
                
                token = tokens[idx]
                clean_token = re.sub(r"[^a-zA-Zа-яА-Я0-9іїєґІЇЄҐ']+", "", token)
                if not clean_token: continue
                
                word_width = word_widths[i]
                
                # Word Timing
                w_start = token_times[idx][0]
                w_end = token_times[idx][1]
                
                # Relative times in ms for tags
                rel_start = int((w_start - t_start) * 1000)
                rel_end = int((w_end - t_start) * 1000)
                
                prefix = ""
                suffix = ""
                
                # --- BACKGROUND (Box) via Vector Drawing ---
                if box_enabled:
                    pad_x = 5
                    pad_y = 2
                    
                    b_w = int(word_width + pad_x * 2)
                    b_h = int(fontsize + pad_y * 2)
                    
                    # Box Center X = current_x + word_width / 2
                    # [RESTORED] +150px offset from "Morning Version"
                    box_center_x = int(current_x + word_width / 2 + 150)
                    
                    box_data = {
                        'x': box_center_x,
                        'w': b_w,
                        'h': b_h,
                        'line_idx': line_idx,
                        'word_idx_in_line': i,
                        'abs_start': w_start,
                        'abs_end': w_end,
                        'opacity': box_opacity_hex,
                        'color': box_color,
                        'prev_x': prev_box_x
                    }
                    page_boxes.append(box_data)
                    prev_box_x = box_center_x
                
                # Apply Effects
                if karaoke:
                    # Karaoke: Start with normal color, highlight during word, return to normal
                    # Initial state: normal color
                    normal_color = style_settings.get('fontcolor', '&H00FFFFFF')
                    
                    # At word start: transition to highlight
                    # At word end: transition back to normal
                    prefix += f"{{\\1c{normal_color}}}{{\\t({rel_start},{rel_start},\\1c{highlight_color})}}{{\\t({rel_end},{rel_end},\\1c{normal_color})}}"

                if animation:
                    # Animation: Start at 100%, scale up to 105% during word, back to 100%
                    # Initial state: normal scale
                    mid = (rel_start + rel_end) // 2
                    prefix += f"{{\\fscx100\\fscy100}}{{\\t({rel_start},{mid},\\fscx105\\fscy105)}}{{\\t({mid},{rel_end},\\fscx100\\fscy100)}}"

                line_parts.append(f"{prefix}{clean_token.upper()}{suffix}")
                
                current_x += word_width + space_width
            
            if line_parts:
                lines_text.append(" ".join(line_parts))
        
        if lines_text:
            full_text = r"\N".join(lines_text)
            events.append({'start': t_start, 'end': t_end, 'fg': full_text, 'boxes': page_boxes})
    
    # [FIX] Prevent event overlaps
    for i in range(len(events) - 1):
        if events[i]['end'] > events[i+1]['start']:
            events[i]['end'] = events[i+1]['start']
        if events[i]['end'] <= events[i]['start']:
            events[i]['end'] = events[i]['start'] + 0.1

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
    
    for ev in events:
        # ev is a dict: {'start', 'end', 'fg', 'boxes': []}
        t0 = ev['start']
        t1 = ev['end']
        text_fg = ev['fg']
        boxes = ev.get('boxes', [])
        
        # Dynamic Vertical Centering based on actual line count
        num_lines = text_fg.count(r"\N") + 1
        max_lines_setting = style_settings.get('max_lines', 1)
        
        # Calculate base position
        current_y = y_pos
        
        # If fewer lines than max, shift to center vertically
        if num_lines < max_lines_setting:
            # Calculate the total vertical space for max_lines
            # Line spacing is 1.2 * fontsize between lines
            # Total height = (max_lines - 1) * line_spacing
            # We want to center num_lines within this total height
            
            # Total height that would be occupied by max_lines
            max_height = (max_lines_setting - 1) * (fontsize * 1.2)
            
            # Actual height occupied by num_lines
            actual_height = (num_lines - 1) * (fontsize * 1.2)
            
            # Shift up by half the difference to center vertically
            total_shift = (max_height - actual_height) / 2
            current_y = int(y_pos - total_shift)
            
        s_time = ass_time(t0)
        e_time = ass_time(t1)
        
        # Layer 0: Background Boxes
        for box in boxes:
            line_idx = box['line_idx']
            
            # Calculate Y for this line
            # Line spacing 1.2
            line_y = current_y - (num_lines - 1 - line_idx) * (fontsize * 1.2)
            box_y = int(line_y + fontsize * 0.5)
            
            bx = box['x']
            bw = box['w']
            bh = box['h']
            
            # Use ABSOLUTE timestamps for this box
            box_start = ass_time(box['abs_start'])
            box_end = ass_time(box['abs_end'])
            
            # Vector Rect (Centered at 0,0)
            rect = create_rounded_rect_path(bw, bh, 15)
            
            # [CRITICAL] Use \c for vector fill color
            style = f"\\c{box['color']}\\bord0\\shad0\\blur0\\alpha&HFF&"
            
            # Animation: Sliding
            anim = f"{{\\alpha{box['opacity']}}}"
            duration_ms = int((box['abs_end'] - box['abs_start']) * 1000)
            
            # Sliding: Use \move if there's a previous box
            if box['prev_x'] is not None and box['word_idx_in_line'] > 0:
                prev_x = box['prev_x']
                prev_y = box_y
                slide_duration = min(150, duration_ms // 2)
                pos = f"\\move({prev_x},{prev_y},{bx},{box_y},0,{slide_duration})"
            else:
                pos = f"\\pos({bx},{box_y})"
            
            ass_line = f"Dialogue: 0,{box_start},{box_end},Default,,0000,0000,0000,,{{{pos}}}{{\\p1}}{style}{anim}{rect}{{\\p0}}"
            ass.append(ass_line)
            
        # Layer 1: Foreground (Text)
        ass_line = f"Dialogue: 1,{s_time},{e_time},Default,,0000,0000,0000,,{{\\pos({x_pos},{current_y})}}{text_fg}"
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

def get_video_resolution(file_path):
    """Returns video width and height."""
    try:
        ff = find_ffmpeg()
        cmd = [ff, "-i", file_path]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, encoding='utf-8')
        # Search for resolution like "1920x1080" or "1080x1920"
        match = re.search(r"(\d{3,4})x(\d{3,4})", result.stderr)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width, height
    except Exception as e:
        log.error(f"Error getting resolution: {e}")
    return None, None

def compress_video(input_path, target_size_mb=49.0):
    """
    Compresses video using intelligent two-pass strategy:
    1. First attempt: Compress at original resolution (Full HD if possible)
    2. If result > 49MB: Compress again at 720p
    
    This preserves quality for videos that can be compressed without resolution reduction.
    """
    log.info(f"Compressing {input_path} to {target_size_mb}MB...")
    
    duration = get_video_duration(input_path)
    if duration <= 0:
        log.error("Could not determine duration for compression.")
        return input_path # Return original if fail

    width, height = get_video_resolution(input_path)
    
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
    
    # PASS 1: Try compressing at original resolution (preserve Full HD)
    log.info("Pass 1: Attempting compression at original resolution...")
    out_path_hd = os.path.join(dir_name, f"{base_name}_compressed_hd.mp4")
    
    cmd_hd = [
        ff, "-y", "-i", input_path,
        "-c:v", "libx264",
        "-b:v", f"{video_bitrate_kbit}k",
        "-maxrate", f"{video_bitrate_kbit * 1.5}k",
        "-bufsize", f"{video_bitrate_kbit * 2}k",
        "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k",
        out_path_hd
    ]
    
    subprocess.run(cmd_hd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Check if HD version fits within limit
    if os.path.exists(out_path_hd):
        hd_size_mb = os.path.getsize(out_path_hd) / (1024 * 1024)
        log.info(f"Pass 1 result: {hd_size_mb:.2f}MB")
        
        if hd_size_mb <= target_size_mb:
            # Success! HD version fits
            log.info(f"✅ Full HD compression successful: {hd_size_mb:.2f}MB")
            return out_path_hd
        else:
            # HD version too large, need to reduce resolution
            log.info(f"⚠️ HD version too large ({hd_size_mb:.2f}MB), reducing to 720p...")
            # Clean up HD attempt
            try:
                os.remove(out_path_hd)
            except:
                pass
    
    # PASS 2: Compress with 720p resolution
    log.info("Pass 2: Compressing with 720p resolution...")
    out_path_720 = os.path.join(dir_name, f"{base_name}_compressed.mp4")
    
    # Determine scale filter for 720p
    scale_filter = ""
    if width and height:
        if width > height:  # Landscape
            scale_filter = "scale=-2:720"
        else:  # Portrait or square
            scale_filter = "scale=720:-2"
        log.info(f"Scaling from {width}x{height} to 720p")
    
    cmd_720 = [
        ff, "-y", "-i", input_path,
        "-c:v", "libx264",
        "-b:v", f"{video_bitrate_kbit}k",
        "-maxrate", f"{video_bitrate_kbit * 1.5}k",
        "-bufsize", f"{video_bitrate_kbit * 2}k",
        "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k"
    ]
    
    # Add scale filter
    if scale_filter:
        cmd_720.insert(4, "-vf")
        cmd_720.insert(5, scale_filter)
    
    cmd_720.append(out_path_720)
    
    subprocess.run(cmd_720, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(out_path_720):
        size_720_mb = os.path.getsize(out_path_720) / (1024 * 1024)
        log.info(f"✅ 720p compression complete: {size_720_mb:.2f}MB")
        return out_path_720
    
    # If both failed, return original
    return input_path
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

        # [FIX] Keep punctuation! User needs it in chat, and layout needs it for sentence breaks
        clean_text = original_text 

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
        context.user_data['style_highlight_color_name'] = saved_settings.get('highlight_color_name', 'Жовтий')
        context.user_data['style_highlight_color_value'] = saved_settings.get('highlight_color_value', '&H0000FFFF')

        # [!!! РОЗБИТТЯ ДОВГОГО ТЕКСТУ !!!]
        await message.reply_text(
            f"Ось розпізнаний текст з пунктуацією.\n\n"
            f"👉 Надішліть виправлений текст (подвійний пробіл - новий рядок) або 'ОК'.\n\n"
            f"📋 Текст для копіювання нижче 👇",
            parse_mode='Markdown'
        )
        
        if len(clean_text) > 4000:
            for i in range(0, len(clean_text), 4000):
                await message.reply_text(f"```\n{clean_text[i:i+4000]}\n```", parse_mode='Markdown')
        else:
            await message.reply_text(f"```\n{clean_text}\n```", parse_mode='Markdown')
        
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
            'karaoke': context.user_data.get('style_karaoke', False),
            'highlight_color_value': context.user_data.get('style_highlight_color_value', '&H0000FFFF')
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

# Highlight colors for karaoke (optimized for visibility)
HIGHLIGHT_COLORS = {
    'Жовтий': ('&H0000FFFF', '🟨'),
    'Червоний': ('&H000000FF', '🟥'),
    'Блакитний': ('&H00FFFF00', '🟦'),
    'Зелений': ('&H0000FF00', '🟩'),
    'Рожевий': ('&H00FF00FF', '🟪'),
    'Помаранчевий': ('&H000099FF', '🟧'),
    'Фіолетовий': ('&H00FF0080', '🟣'),
    'Лаймовий': ('&H0000FF99', '💚'),
    'Золотий': ('&H0000D7FF', '🔶'),
    'Білий': ('&H00FFFFFF', '⬜'),
    'Салатовий': ('&H0066FF99', '🟢'),
}
HIGHLIGHT_COLOR_NAMES = list(HIGHLIGHT_COLORS.keys())

# Font colors with emojis
STYLE_COLORS_EMOJI = {
    'Білий': ('&H00FFFFFF', '⬜'),
    'Жовтий': ('&H0000FFFF', '🟨'),
    'Червоний': ('&H000000FF', '🟥'),
    'Зелений': ('&H0000FF00', '🟩'),
    'Синій': ('&H00FF0000', '🟦'),
}

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
    highlight_color = user_data.get('style_highlight_color_name', 'Жовтий')
    
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
        # Create color buttons
        color_buttons = []
        for cname, (cval, emoji) in STYLE_COLORS_EMOJI.items():
            color_buttons.append(InlineKeyboardButton(emoji, callback_data=f'pick_color_{cname}'))
        
        keyboard = [
            [InlineKeyboardButton("Шрифт ›", callback_data='set_font_next')],
            [InlineKeyboardButton("- Розмір", callback_data='set_size_minus'),
             InlineKeyboardButton("+ Розмір", callback_data='set_size_plus')],
            color_buttons,
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
            f"Колір підсвітки: {highlight_color}\n"
            f"_(Караоке змінює колір активного слова)_"
        )
        # Create highlight color buttons - split into 2 rows
        highlight_btns = []
        for cname, (cval, emoji) in HIGHLIGHT_COLORS.items():
            highlight_btns.append(InlineKeyboardButton(emoji, callback_data=f'pick_highlight_{cname}'))
        
        # Split into rows of 6
        row1 = highlight_btns[:6]
        row2 = highlight_btns[6:]
        
        keyboard = [
            [InlineKeyboardButton(f"Анімація {anim}", callback_data='toggle_anim')],
            [InlineKeyboardButton(f"Караоке {karaoke}", callback_data='toggle_karaoke')],
            row1,
            row2,
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

    # Font/Size Setters
    elif data == 'set_size_plus':
        current_size = user_data.get('style_fontsize', 93)
        user_data['style_fontsize'] = min(current_size + 5, 200)
        log.info(f"Font size increased to {user_data['style_fontsize']}")
    elif data == 'set_size_minus':
        current_size = user_data.get('style_fontsize', 93)
        user_data['style_fontsize'] = max(current_size - 5, 30)
        log.info(f"Font size decreased to {user_data['style_fontsize']}")
    elif data == 'set_font_next':
        current = user_data.get('style_font_name', STYLE_FONTS[0])
        idx = STYLE_FONTS.index(current) if current in STYLE_FONTS else 0
        user_data['style_font_name'] = STYLE_FONTS[(idx + 1) % len(STYLE_FONTS)]

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
    
    # Direct color pickers
    elif data.startswith('pick_color_'):
        color_name = data.replace('pick_color_', '')
        log.info(f"Color picker triggered: {color_name}, available: {list(STYLE_COLORS_EMOJI.keys())}")
        if color_name in STYLE_COLORS_EMOJI:
            user_data['style_color_name'] = color_name
            user_data['style_color_value'] = STYLE_COLORS_EMOJI[color_name][0]
            log.info(f"Set color to {color_name}: {STYLE_COLORS_EMOJI[color_name][0]}")
        else:
            log.warning(f"Color {color_name} not found in STYLE_COLORS_EMOJI")
    elif data.startswith('pick_highlight_'):
        color_name = data.replace('pick_highlight_', '')
        log.info(f"Highlight picker triggered: {color_name}, available: {list(HIGHLIGHT_COLORS.keys())}")
        if color_name in HIGHLIGHT_COLORS:
            user_data['style_highlight_color_name'] = color_name
            user_data['style_highlight_color_value'] = HIGHLIGHT_COLORS[color_name][0]
            log.info(f"Set highlight to {color_name}: {HIGHLIGHT_COLORS[color_name][0]}")
        else:
            log.warning(f"Highlight color {color_name} not found in HIGHLIGHT_COLORS")

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
        'karaoke': user_data.get('style_karaoke'),
        'highlight_color_name': user_data.get('style_highlight_color_name'),
        'highlight_color_value': user_data.get('style_highlight_color_value')
    }
    save_settings(query.message.chat_id, current_settings)

    # Refresh current menu
    # Need to know which menu we are in. 
    # A simple hack is to check the button that was clicked or store state.
    # But since we don't store menu state in user_data, we can infer or just default to 'main' 
    # or try to guess. 
    # Better: Pass the menu state in callback data? e.g. 'set_size_plus:style'
    # For now, simple heuristic based on data prefix
    next_menu = 'main'
    if data.startswith('set_size') or data.startswith('set_color') or data.startswith('set_font') or data.startswith('toggle_shadow') or data.startswith('toggle_outline') or data.startswith('pick_color_'):
        next_menu = 'style'
    elif data.startswith('set_wpl') or data.startswith('set_lines') or data.startswith('set_margin'):
        next_menu = 'layout'
    elif data.startswith('toggle_anim') or data.startswith('toggle_karaoke') or data.startswith('set_highlight') or data.startswith('pick_highlight_'):
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
    max_retries = 30  # Wait up to 2.5 minutes
    
    log.info("Checking network connectivity...")
    import httpx
    import socket
    
    while retries < max_retries:
        try:
            # Try DNS resolution first
            try:
                socket.gethostbyname("api.telegram.org")
            except socket.gaierror:
                log.warning(f"DNS lookup failed ({retries+1}/{max_retries})")
                
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.get(url)
                log.info("Network is UP! 🚀")
                return
        except Exception as e:
            retries += 1
            log.warning(f"Network check failed ({retries}/{max_retries}): {e}")
            await asyncio.sleep(5)  # Wait 5 seconds between tries
            
    log.error("Network check failed after max retries. Proceeding anyway...")


def main():
    """Main function to start the bot"""
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
                CallbackQueryHandler(handle_settings_callback, pattern='^(menu_|set_|toggle_|pick_|process_|new_video)')
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
        # Webhook mode for Render/Hugging Face
        log.info(f"Running in webhook mode on port {PORT}")
        log.info("Starting bot...")
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path="",
            webhook_url=RENDER_EXTERNAL_URL
        )
    else:
        log.info("Running in polling mode...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
