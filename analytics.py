# -*- coding: utf-8 -*-
"""
Analytics module for tracking bot usage
Сте, ваш Chat ID: 1236683290
"""
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

log = logging.getLogger(__name__)

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Creates and returns a database connection"""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set!")
    return psycopg2.connect(DATABASE_URL, sslmode='require')

# ====================================================================
# ANALYTICS TABLE INITIALIZATION
# ====================================================================

def init_analytics_table() -> bool:
    """
    Creates the analytics table if it doesn't exist.
    Called during bot initialization.
    """
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        log.warning("Database not available for analytics")
        return False
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create analytics table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS usage_analytics (
                id SERIAL PRIMARY KEY,
                chat_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_chat_id 
            ON usage_analytics(chat_id)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_created_at 
            ON usage_analytics(created_at)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_event_type 
            ON usage_analytics(event_type)
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
        log.info("✅ Analytics table initialized")
        print("✅ Analytics table initialized")
        return True
        
    except Exception as e:
        log.error(f"❌ Error initializing analytics table: {e}")
        return False

# ==================================================================== 
# EVENT LOGGING
# ====================================================================

def log_event(chat_id: str, event_type: str, event_data: Dict = None) -> bool:
    """
    Logs an event to the analytics table.
    
    Event types:
    - 'user_started' - New user started bot
    - 'video_uploaded' - Video received
    - 'video_processed' - Video processing completed
    - 'settings_changed' - User changed settings
    - 'error_occurred' - Error during processing
    
    Args:
        chat_id: Telegram chat ID
        event_type: Type of event
        event_data: Optional dictionary with event details
    
    Returns:
        True if successful
    """
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        return False
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO usage_analytics (chat_id, event_type, event_data)
            VALUES (%s, %s, %s)
        """, (str(chat_id), event_type, Json(event_data) if event_data else None))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        log.error(f"❌ Error logging analytics event: {e}")
        return False

# ====================================================================
# STATISTICS RETRIEVAL
# ====================================================================

def get_stats_today() -> Dict:
    """Returns analytics for today"""
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        return {"error": "Database not available"}
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        today = datetime.now().date()
        
        # General stats
        cur.execute("""
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT chat_id) as unique_users
            FROM usage_analytics
            WHERE DATE(created_at) = %s
        """, (today,))
        
        general_stats = cur.fetchone()
        
        # Events by type
        cur.execute("""
            SELECT 
                event_type,
                COUNT(*) as count
            FROM usage_analytics
            WHERE DATE(created_at) = %s
            GROUP BY event_type
            ORDER BY count DESC
        """, (today,))
        
        events_by_type = cur.fetchall()
        
        # Hourly distribution
        cur.execute("""
            SELECT 
                EXTRACT(HOUR FROM created_at)::INTEGER as hour,
                COUNT(*) as count
            FROM usage_analytics
            WHERE DATE(created_at) = %s
            GROUP BY hour
            ORDER BY hour
        """, (today,))
        
        hourly = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {
            "period": "today",
            "date": str(today),
            "general": dict(general_stats) if general_stats else {},
            "events_by_type": [dict(row) for row in events_by_type],
            "hourly_distribution": [dict(row) for row in hourly]
        }
        
    except Exception as e:
        log.error(f"❌ Error getting today stats: {e}")
        return {"error": str(e)}

def get_stats_week() -> Dict:
    """Returns analytics for the past 7 days"""
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        return {"error": "Database not available"}
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        week_ago = datetime.now() - timedelta(days=7)
        
        # General stats
        cur.execute("""
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT chat_id) as unique_users
            FROM usage_analytics
            WHERE created_at >= %s
        """, (week_ago,))
        
        general_stats = cur.fetchone()
        
        # Events by type
        cur.execute("""
            SELECT 
                event_type,
                COUNT(*) as count
            FROM usage_analytics
            WHERE created_at >= %s
            GROUP BY event_type
            ORDER BY count DESC
        """, (week_ago,))
        
        events_by_type = cur.fetchall()
        
        # Daily distribution
        cur.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as count
            FROM usage_analytics
            WHERE created_at >= %s
            GROUP BY date
            ORDER BY date
        """, (week_ago,))
        
        daily = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {
            "period": "week",
            "start_date": str(week_ago.date()),
            "general": dict(general_stats) if general_stats else {},
            "events_by_type": [dict(row) for row in events_by_type],
            "daily_distribution": [{"date": str(row['date']), "count": row['count']} for row in daily]
        }
        
    except Exception as e:
        log.error(f"❌ Error getting week stats: {e}")
        return {"error": str(e)}

def get_stats_month() -> Dict:
    """Returns analytics for the past 30 days"""
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        return {"error": "Database not available"}
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        month_ago = datetime.now() - timedelta(days=30)
        
        # General stats
        cur.execute("""
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT chat_id) as unique_users
            FROM usage_analytics
            WHERE created_at >= %s
        """, (month_ago,))
        
        general_stats = cur.fetchone()
        
        # Events by type
        cur.execute("""
            SELECT 
                event_type,
                COUNT(*) as count
            FROM usage_analytics
            WHERE created_at >= %s
            GROUP BY event_type
            ORDER BY count DESC
        """, (month_ago,))
        
        events_by_type = cur.fetchall()
        
        # Daily distribution
        cur.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as count
            FROM usage_analytics
            WHERE created_at >= %s
            GROUP BY date
            ORDER BY date
        """, (month_ago,))
        
        daily = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {
            "period": "month",
            "start_date": str(month_ago.date()),
            "general": dict(general_stats) if general_stats else {},
            "events_by_type": [dict(row) for row in events_by_type],
            "daily_distribution": [{"date": str(row['date']), "count": row['count']} for row in daily]
        }
        
    except Exception as e:
        log.error(f"❌ Error getting month stats: {e}")
        return {"error": str(e)}

def get_top_users(period='today', limit=10) -> List[Dict]:
    """
    Returns top users by activity.
    
    Args:
        period: 'today', 'week', or 'month'
        limit: Max number of users
    
    Returns:
        List of user activity data
    """
    if not DATABASE_URL or not POSTGRES_AVAILABLE:
        return []
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Calculate date filter
        if period == 'today':
            filter_date = datetime.now().date()
            date_condition = "DATE(created_at) = %s"
        elif period == 'week':
            filter_date = datetime.now() - timedelta(days=7)
            date_condition = "created_at >= %s"
        else:  # month
            filter_date = datetime.now() - timedelta(days=30)
            date_condition = "created_at >= %s"
        
        cur.execute(f"""
            SELECT 
                chat_id,
                COUNT(*) as activity_count,
                MAX(created_at) as last_activity
            FROM usage_analytics
            WHERE {date_condition}
            GROUP BY chat_id
            ORDER BY activity_count DESC
            LIMIT %s
        """, (filter_date, limit))
        
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        log.error(f"❌ Error getting top users: {e}")
        return []
