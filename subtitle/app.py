#!/usr/bin/env python3
"""
DIAGNOSTIC: Show ALL environment variables
"""
import os

print("=" * 80)
print("DIAGNOSTIC: ALL ENVIRONMENT VARIABLES")
print("=" * 80)

# Show all env vars
for key, value in sorted(os.environ.items()):
    # Hide sensitive values (show only first 10 chars)
    if len(value) > 20:
        display_value = value[:10] + "..." + f"(len={len(value)})"
    else:
        display_value = value
    print(f"{key} = {display_value}")

print("\n" + "=" * 80)
print("CHECKING REQUIRED SECRETS:")
print("=" * 80)

bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
groq_key = os.getenv('GROQ_API_KEY')

print(f"\n1. TELEGRAM_BOT_TOKEN: {'✅ FOUND' if bot_token else '❌ NOT FOUND'}")
if bot_token:
    print(f"   Length: {len(bot_token)} chars")
    print(f"   Starts with: {bot_token[:15]}...")

print(f"\n2. GROQ_API_KEY: {'✅ FOUND' if groq_key else '❌ NOT FOUND'}")
if groq_key:
    print(f"   Length: {len(groq_key)} chars")
    print(f"   Starts with: {groq_key[:10]}...")

print("\n" + "=" * 80)

if not bot_token or not groq_key:
    print("\n❌ SECRETS ARE MISSING!")
    print("\n📝 TO FIX THIS:")
    print("1. Go to: https://huggingface.co/spaces/StarBro/subritle/settings")
    print("2. Find section: 'Variables and secrets'")
    print("3. Click 'New secret'")
    print("4. Add these TWO secrets:")
    print("   - Name: TELEGRAM_BOT_TOKEN")
    print("     Value: <your bot token from @BotFather>")
    print("   - Name: GROQ_API_KEY")
    print("     Value: <your Groq API key>")
    print("5. RESTART the Space")
    print("\n" + "=" * 80)
    import sys
    sys.exit(1)

print("\n✅ ALL SECRETS FOUND! Starting bot...")
print("Attempting to import bot module...")

try:
    import bot
    print("✅ Bot module imported successfully!")
    print("Starting bot.main()...")
    bot.main()
except Exception as e:
    print(f"\n❌ FAILED TO START BOT!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    import sys
    sys.exit(1)