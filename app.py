#!/usr/bin/env python3
"""
SUPER SIMPLE TEST - just check if secrets exist
"""
import os
import sys

print("=" * 60)
print("TESTING HUGGING FACE SECRETS")
print("=" * 60)

# Check environment variables
bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
groq_key = os.getenv('GROQ_API_KEY')

print(f"\nTELEGRAM_BOT_TOKEN exists: {bool(bot_token)}")
if bot_token:
    print(f"  Value starts with: {bot_token[:10]}...")
else:
    print("  ❌ NOT FOUND! Add it in Space Settings → Repository secrets")

print(f"\nGROQ_API_KEY exists: {bool(groq_key)}")
if groq_key:
    print(f"  Value starts with: {groq_key[:10]}...")
else:
    print("  ❌ NOT FOUND! Add it in Space Settings → Repository secrets")

print("\n" + "=" * 60)

if not bot_token or not groq_key:
    print("⚠️  SECRETS MISSING - Bot cannot start!")
    print("\nTO FIX:")
    print("1. Go to https://huggingface.co/spaces/StarBro/subritle/settings")
    print("2. Scroll to 'Repository secrets'")
    print("3. Add TELEGRAM_BOT_TOKEN and GROQ_API_KEY")
    print("4. Restart the Space")
    sys.exit(1)

print("✅ All secrets found! Starting bot...")

# Only import bot if secrets are OK
import bot
