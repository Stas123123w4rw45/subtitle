# Subtitle Bot

Telegram bot for generating subtitles with animated word highlights.

## Features

- **Automatic Transcription**: Uses Groq API (Whisper) for fast and accurate transcription.
- **Animated Subtitles**: Generates video with animated subtitles.
- **Word Highlight**: Active words are highlighted with a rounded background animation.
- **Customization**: Adjustable font size, color, font face, and bottom margin.

## Setup

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Install FFmpeg.
4.  Set environment variables:
    - `TELEGRAM_BOT_TOKEN`: Your Telegram Bot Token.
    - `GROQ_API_KEY`: Your Groq API Key.

## Usage

1.  Run the bot:
    ```bash
    python bot.py
    ```
2.  Send a video or audio file to the bot on Telegram.
3.  The bot will transcribe the audio and ask for confirmation/editing of the text.
4.  Customize the style using the inline keyboard.
5.  The bot will send back the video with burned-in subtitles.

## New Feature: Animated Word Highlight

The bot now supports a "WordHighlight" style. Each active word is individually highlighted with a rounded background effect that animates smoothly. This is achieved using advanced ASS subtitle styling.
