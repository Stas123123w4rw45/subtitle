# 💾 Альтернативні Методи Збереження Налаштувань

## Якщо Railway Volume недоступний

### Варіант 1: Railway PostgreSQL (Рекомендовано)

Railway надає безкоштовну PostgreSQL базу даних.

#### Крок 1: Додайте PostgreSQL до проекту

1. У Railway Dashboard → Ваш проект
2. Натисніть **"+ New"** → **"Database"** → **"Add PostgreSQL"**
3. Railway автоматично створить змінну `DATABASE_URL`

#### Крок 2: Оновіть `requirements.txt`

Додайте:
```
psycopg2-binary
```

#### Крок 3: Створіть файл `database.py`

```python
import os
import json
import psycopg2
from psycopg2.extras import Json

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Створює таблицю для налаштувань"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            chat_id TEXT PRIMARY KEY,
            settings JSONB NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def load_settings(chat_id):
    """Завантажує налаштування користувача"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT settings FROM user_settings WHERE chat_id = %s",
            (str(chat_id),)
        )
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result:
            return result[0]
        return {}
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}

def save_settings(chat_id, settings):
    """Зберігає налаштування користувача"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_settings (chat_id, settings)
            VALUES (%s, %s)
            ON CONFLICT (chat_id) 
            DO UPDATE SET settings = EXCLUDED.settings,
                         updated_at = CURRENT_TIMESTAMP
        """, (str(chat_id), Json(settings)))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error saving settings: {e}")
```

#### Крок 4: Оновіть `bot.py`

Замініть функції `load_settings` і `save_settings` на:

```python
from database import init_db, load_settings, save_settings

# В main():
init_db()  # Ініціалізувати базу даних при старті
```

---

### Варіант 2: Railway Redis (Швидше)

Той самий підхід, але з Redis для кешування.

#### Крок 1: Додайте Redis

1. Railway Dashboard → **"+ New"** → **"Database"** → **"Add Redis"**
2. Railway створить `REDIS_URL`

#### Крок 2: Оновіть `requirements.txt`

```
redis
```

#### Крок 3: Створіть `redis_storage.py`

```python
import os
import json
import redis

REDIS_URL = os.getenv("REDIS_URL")
r = redis.from_url(REDIS_URL)

def load_settings(chat_id):
    """Завантажує налаштування з Redis"""
    try:
        data = r.get(f"settings:{chat_id}")
        if data:
            return json.loads(data)
        return {}
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}

def save_settings(chat_id, settings):
    """Зберігає налаштування в Redis"""
    try:
        r.set(f"settings:{chat_id}", json.dumps(settings, ensure_ascii=False))
    except Exception as e:
        print(f"Error saving settings: {e}")
```

---

### Варіант 3: Environment Variables (Тимчасове рішення)

**Увага**: Цей метод зберігає налаштування тільки для одного користувача.

Додайте в Railway Environment Variables:

```
DEFAULT_FONT_NAME=Peace Sans
DEFAULT_FONT_SIZE=93
DEFAULT_COLOR_NAME=Білий
DEFAULT_COLOR_VALUE=&H00FFFFFF
```

Та використовуйте їх як дефолтні значення в коді.

---

### Варіант 4: GitHub Gist (Експериментальний)

Зберігайте налаштування в приватному GitHub Gist.

**Переваги**: Безкоштовно, версіонування
**Недоліки**: Повільніше, потрібен токен

---

## 🎯 Рекомендація

| Метод | Швидкість | Надійність | Складність | Ціна |
|-------|-----------|------------|------------|------|
| **Railway Volume** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | $ (платно?) |
| **PostgreSQL** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Безкоштовно |
| **Redis** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Безкоштовно |
| **Env Vars** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | Безкоштовно |
| **GitHub Gist** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Безкоштовно |

### Що обрати?

1. **Спочатку спробуйте Volume** (найпростіше)
2. Якщо не працює → **PostgreSQL** (найнадійніше)
3. Для швидкості → **Redis** (найшвидше)

---

**Примітка**: Всі методи повністю сумісні з вашим поточним кодом - потрібно тільки замінити функції `load_settings()` і `save_settings()`.
