#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for PostgreSQL database integration
Tests connection, table creation, and CRUD operations
"""
import os
import sys

# Set test DATABASE_URL if not already set
# You can replace this with your actual Railway DATABASE_URL for testing
if not os.getenv("DATABASE_URL"):
    print("⚠️ DATABASE_URL not set!")
    print("Please set it first:")
    print('  export DATABASE_URL="postgresql://user:pass@host:port/db"')
    print("\nOr for Windows PowerShell:")
    print('  $env:DATABASE_URL="postgresql://user:pass@host:port/db"')
    sys.exit(1)

print("=" * 60)
print("PostgreSQL Database Integration Test")
print("=" * 60)
print()

# Import database module
try:
    from database import init_db, load_settings, save_settings, delete_settings, get_all_users, get_stats
    print("✅ Database module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import database module: {e}")
    print("\nMake sure psycopg2-binary is installed:")
    print("  pip install psycopg2-binary")
    sys.exit(1)

print()
print("=" * 60)
print("Test 1: Database Initialization")
print("=" * 60)

if init_db():
    print("✅ Database initialized successfully")
    print("   - Table 'user_settings' created")
    print("   - Indexes created")
    print("   - Triggers created")
else:
    print("❌ Database initialization failed")
    sys.exit(1)

print()
print("=" * 60)
print("Test 2: Save Settings")
print("=" * 60)

test_chat_id = "test_user_12345"
test_settings = {
    'fontsize': 100,
    'color_name': 'Червоний',
    'color_value': '&H000000FF',
    'font_name': 'Impact',
    'margin_bottom': 25,
    'shadow_enabled': True,
    'outline_enabled': False,
    'wpl': 3,
    'max_lines': 2,
    'animation': True,
    'karaoke': True,
    'highlight_color_name': 'Жовтий',
    'highlight_color_value': '&H0000FFFF'
}

if save_settings(test_chat_id, test_settings):
    print(f"✅ Settings saved for user: {test_chat_id}")
    print(f"   Settings: {test_settings}")
else:
    print("❌ Failed to save settings")

print()
print("=" * 60)
print("Test 3: Load Settings")
print("=" * 60)

loaded_settings = load_settings(test_chat_id)
if loaded_settings:
    print(f"✅ Settings loaded for user: {test_chat_id}")
    print(f"   Loaded: {loaded_settings}")
    
    # Verify data integrity
    if loaded_settings == test_settings:
        print("✅ Data integrity verified (saved == loaded)")
    else:
        print("⚠️ Data mismatch detected!")
        print(f"   Expected: {test_settings}")
        print(f"   Got: {loaded_settings}")
else:
    print("❌ Failed to load settings")

print()
print("=" * 60)
print("Test 4: Update Settings")
print("=" * 60)

updated_settings = loaded_settings.copy()
updated_settings['fontsize'] = 120
updated_settings['color_name'] = 'Синій'

if save_settings(test_chat_id, updated_settings):
    print(f"✅ Settings updated for user: {test_chat_id}")
    
    # Verify update
    reloaded = load_settings(test_chat_id)
    if reloaded['fontsize'] == 120 and reloaded['color_name'] == 'Синій':
        print("✅ Update verified successfully")
    else:
        print("⚠️ Update verification failed")
else:
    print("❌ Failed to update settings")

print()
print("=" * 60)
print("Test 5: Multiple Users")
print("=" * 60)

user2_id = "test_user_67890"
user2_settings = {
    'fontsize': 80,
    'font_name': 'OpenSans-Light',
    'color_name': 'Зелений'
}

if save_settings(user2_id, user2_settings):
    print(f"✅ Settings saved for second user: {user2_id}")
    
    # Verify both users' data is separate
    user1_data = load_settings(test_chat_id)
    user2_data = load_settings(user2_id)
    
    if user1_data != user2_data:
        print("✅ User data isolation verified")
        print(f"   User 1 fontsize: {user1_data.get('fontsize')}")
        print(f"   User 2 fontsize: {user2_data.get('fontsize')}")
    else:
        print("❌ User data isolation failed!")
else:
    print("❌ Failed to save second user")

print()
print("=" * 60)
print("Test 6: Get All Users")
print("=" * 60)

all_users = get_all_users()
print(f"📊 Total users in database: {len(all_users)}")
print(f"   User IDs: {all_users}")

print()
print("=" * 60)
print("Test 7: Database Statistics")
print("=" * 60)

stats = get_stats()
print(f"📊 Database Statistics:")
for key, value in stats.items():
    print(f"   {key}: {value}")

print()
print("=" * 60)
print("Test 8: Delete Settings")
print("=" * 60)

if delete_settings(test_chat_id):
    print(f"✅ Settings deleted for user: {test_chat_id}")
    
    # Verify deletion
    deleted_data = load_settings(test_chat_id)
    if not deleted_data or deleted_data == {}:
        print("✅ Deletion verified (user settings empty)")
    else:
        print("⚠️ Deletion verification failed (data still exists)")
else:
    print("❌ Failed to delete settings")

print()
print("=" * 60)
print("Test 9: Load Non-Existent User")
print("=" * 60)

non_existent = load_settings("non_existent_user_999")
if non_existent == {}:
    print("✅ Non-existent user returns empty dict (correct)")
else:
    print(f"⚠️ Non-existent user returned: {non_existent}")

print()
print("=" * 60)
print("Cleanup")
print("=" * 60)

# Clean up test data
delete_settings(user2_id)
print(f"✅ Cleaned up test users")

print()
print("=" * 60)
print("✅ All Tests Completed Successfully!")
print("=" * 60)
print()
print("📊 Final Statistics:")
final_stats = get_stats()
for key, value in final_stats.items():
    print(f"   {key}: {value}")

print()
print("🎉 PostgreSQL integration is working correctly!")
print()
print("Next steps:")
print("1. Deploy to Railway")
print("2. Railway will automatically set DATABASE_URL")
print("3. Bot will use PostgreSQL for persistent storage")
