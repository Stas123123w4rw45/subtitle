#!/bin/bash

# Show what we have
echo "================================"
echo "Environment Check:"
echo "================================"
echo "TELEGRAM_BOT_TOKEN set: $([[ -n \"$TELEGRAM_BOT_TOKEN\" ]] && echo 'YES'  || echo 'NO')"
echo "GROQ_API_KEY set: $([[ -n \"$GROQ_API_KEY\" ]] && echo 'YES' || echo 'NO')"
echo "================================"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start Bot via app.py (which checks secrets)
echo "Starting Telegram Bot..."
python3 app.py
