#!/usr/bin/env python3
"""
Wrapper to run bot.py on Hugging Face Spaces
Uses webhook mode for better compatibility
"""
import os
import sys

# Set polling mode for HF Spaces (webhook requires public URL)
os.environ['RENDER_EXTERNAL_URL'] = ''

# Import and run the bot
import bot

if __name__ == "__main__":
    print("Starting bot from app.py wrapper...")
