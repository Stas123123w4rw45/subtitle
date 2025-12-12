# -*- coding: utf-8 -*-
"""
Database module for storing user settings in PostgreSQL
"""
import os
import json
import logging
from typing import Dict, Optional

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ psycopg2 not installed. Install with: pip install psycopg2-binary")

log = logging.getLogger(__name__)

# Database URL from Railway environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Creates and returns a database connection"""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set!")
    
    if not POSTGRES_AVAILABLE:
        raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")
    
    return psycopg2.connect(DATABASE_URL, sslmode='require')

def init_db():
    """
    Initializes the database by creating the user_settings table if it doesn't exist.
    Should be called once when the bot starts.
    """
    if not DATABASE_URL:
        log.warning("DATABASE_URL not set. Database not initialized.")
        return False
    
    if not POSTGRES_AVAILABLE:
        log.warning("psycopg2 not available. Database not initialized.")
        return False
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create table for user settings
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                chat_id TEXT PRIMARY KEY,
                settings JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on chat_id for faster lookups
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_settings_chat_id 
            ON user_settings(chat_id)
        """)
        
        # Create trigger to update updated_at timestamp
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        cur.execute("""
            DROP TRIGGER IF EXISTS update_user_settings_updated_at ON user_settings;
        """)
        
        cur.execute("""
            CREATE TRIGGER update_user_settings_updated_at
            BEFORE UPDATE ON user_settings
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
        log.info("✅ Database initialized successfully")
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        log.error(f"❌ Error initializing database: {e}")
        print(f"❌ Error initializing database: {e}")
        return False

def load_settings(chat_id: str) -> Dict:
    """
    Loads user settings from PostgreSQL database.
    
    Args:
        chat_id: Telegram chat ID (will be converted to string)
    
    Returns:
        Dictionary with user settings, or empty dict if not found
    """
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        log.warning("Database not available, returning empty settings")
        return {}
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute(
            "SELECT settings FROM user_settings WHERE chat_id = %s",
            (str(chat_id),)
        )
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result and result['settings']:
            log.info(f"✅ Settings loaded for user {chat_id}")
            return dict(result['settings'])
        
        log.info(f"ℹ️ No settings found for user {chat_id}, returning defaults")
        return {}
        
    except Exception as e:
        log.error(f"❌ Error loading settings for {chat_id}: {e}")
        return {}

def save_settings(chat_id: str, settings: Dict) -> bool:
    """
    Saves user settings to PostgreSQL database.
    Uses UPSERT (INSERT ... ON CONFLICT UPDATE) for atomic operation.
    
    Args:
        chat_id: Telegram chat ID (will be converted to string)
        settings: Dictionary with user settings
    
    Returns:
        True if successful, False otherwise
    """
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        log.warning("Database not available, settings not saved")
        return False
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # UPSERT: Insert or update if exists
        cur.execute("""
            INSERT INTO user_settings (chat_id, settings)
            VALUES (%s, %s)
            ON CONFLICT (chat_id) 
            DO UPDATE SET 
                settings = EXCLUDED.settings,
                updated_at = CURRENT_TIMESTAMP
        """, (str(chat_id), Json(settings)))
        
        conn.commit()
        cur.close()
        conn.close()
        
        log.info(f"✅ Settings saved for user {chat_id}")
        return True
        
    except Exception as e:
        log.error(f"❌ Error saving settings for {chat_id}: {e}")
        return False

def delete_settings(chat_id: str) -> bool:
    """
    Deletes user settings from database.
    
    Args:
        chat_id: Telegram chat ID
    
    Returns:
        True if successful, False otherwise
    """
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        log.warning("Database not available")
        return False
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            "DELETE FROM user_settings WHERE chat_id = %s",
            (str(chat_id),)
        )
        
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        
        if deleted > 0:
            log.info(f"✅ Settings deleted for user {chat_id}")
            return True
        else:
            log.info(f"ℹ️ No settings found for user {chat_id}")
            return False
        
    except Exception as e:
        log.error(f"❌ Error deleting settings for {chat_id}: {e}")
        return False

def get_all_users() -> list:
    """
    Returns list of all chat IDs that have saved settings.
    Useful for analytics or migrations.
    
    Returns:
        List of chat IDs (as strings)
    """
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        return []
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT chat_id FROM user_settings ORDER BY created_at")
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return [row[0] for row in results]
        
    except Exception as e:
        log.error(f"❌ Error getting users: {e}")
        return []

def get_stats() -> Dict:
    """
    Returns database statistics.
    
    Returns:
        Dictionary with stats (total_users, etc.)
    """
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        return {"error": "Database not available"}
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                COUNT(*) as total_users,
                MIN(created_at) as first_user_date,
                MAX(updated_at) as last_update_date
            FROM user_settings
        """)
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        return dict(result) if result else {}
        
    except Exception as e:
        log.error(f"❌ Error getting stats: {e}")
        return {"error": str(e)}

# Fallback to JSON file if database is not available
def load_settings_fallback(chat_id: str) -> Dict:
    """Fallback to JSON file if PostgreSQL is not available"""
    import json
    import os
    
    settings_file = os.getenv("SETTINGS_FILE", "user_settings.json")
    
    try:
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get(str(chat_id), {})
    except Exception as e:
        log.error(f"Error loading fallback settings: {e}")
    
    return {}

def save_settings_fallback(chat_id: str, settings: Dict) -> bool:
    """Fallback to JSON file if PostgreSQL is not available"""
    import json
    import os
    
    settings_file = os.getenv("SETTINGS_FILE", "user_settings.json")
    
    try:
        data = {}
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except:
                    pass
        
        data[str(chat_id)] = settings
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        log.error(f"Error saving fallback settings: {e}")
        return False
