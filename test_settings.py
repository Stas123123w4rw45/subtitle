#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовий скрипт для перевірки збереження налаштувань
"""
import os
import json

# Перевірка шляху до файлу налаштувань
SETTINGS_DIR = "/app/data" if os.path.exists("/app/data") else "."
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "user_settings.json")

print(f"📁 Директорія налаштувань: {SETTINGS_DIR}")
print(f"📄 Файл налаштувань: {SETTINGS_FILE}")
print()

# Перевірка наявності директорії
if os.path.exists(SETTINGS_DIR):
    print(f"✅ Директорія існує: {SETTINGS_DIR}")
else:
    print(f"❌ Директорія НЕ існує: {SETTINGS_DIR}")
    print("   Створюю директорію...")
    try:
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        print("   ✅ Директорію створено")
    except Exception as e:
        print(f"   ❌ Помилка: {e}")

print()

# Перевірка прав на запис
try:
    test_file = os.path.join(SETTINGS_DIR, ".test_write")
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    print("✅ Права на запис є")
except Exception as e:
    print(f"❌ Немає прав на запис: {e}")

print()

# Перевірка наявності файлу налаштувань
if os.path.exists(SETTINGS_FILE):
    print(f"✅ Файл налаштувань існує: {SETTINGS_FILE}")
    try:
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
        print(f"📊 Кількість користувачів: {len(data)}")
        print(f"📊 ID користувачів: {list(data.keys())}")
        print()
        for user_id, settings in data.items():
            print(f"👤 Користувач {user_id}:")
            print(f"   🎨 Шрифт: {settings.get('font_name', 'Не встановлено')}")
            print(f"   📏 Розмір: {settings.get('fontsize', 'Не встановлено')}")
            print(f"   🎨 Колір: {settings.get('color_name', 'Не встановлено')}")
            print()
    except Exception as e:
        print(f"❌ Помилка читання файлу: {e}")
else:
    print(f"⚠️ Файл налаштувань НЕ існує: {SETTINGS_FILE}")
    print("   Це нормально, якщо бот ще не використовувався")

print()
print("=" * 60)
print("🧪 ТЕСТ: Збереження налаштувань")
print("=" * 60)

# Тестове збереження
test_chat_id = "test_user_123"
test_settings = {
    'fontsize': 93,
    'color_name': 'Жовтий',
    'color_value': '&H0000FFFF',
    'font_name': 'Peace Sans',
    'margin_bottom': 30,
    'shadow_enabled': True,
    'outline_enabled': True,
    'wpl': 2,
    'max_lines': 1,
    'animation': False,
    'karaoke': False,
    'highlight_color_name': 'Червоний',
    'highlight_color_value': '&H000000FF'
}

try:
    # Завантаження існуючих дані
    data = {}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            try:
                data = json.load(f)
            except:
                pass
    
    # Збереження тестових даних
    data[test_chat_id] = test_settings
    
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✅ Тестові налаштування збережено")
    
    # Перевірка збереження
    with open(SETTINGS_FILE, 'r') as f:
        loaded_data = json.load(f)
    
    if test_chat_id in loaded_data:
        print("✅ Тестові налаштування успішно завантажено")
        if loaded_data[test_chat_id] == test_settings:
            print("✅ Дані збереглися коректно")
        else:
            print("⚠️ Дані відрізняються від збережених")
    else:
        print("❌ Тестові налаштування не знайдено")
        
except Exception as e:
    print(f"❌ Помилка збереження: {e}")

print()
print("=" * 60)
print("✅ Тест завершено")
print("=" * 60)
