#!/bin/bash

# Check for environment variables
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ Error: TELEGRAM_BOT_TOKEN is not set."
    exit 1
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ Error: GROQ_API_KEY is not set."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start Bot
echo "Starting Telegram Bot..."
sleep 5 # Wait for network/DNS to settle
python3 bot.py
