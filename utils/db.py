import psycopg2
from utils.config import settings

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            dbname=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
        )
        return conn
    except Exception as e:
        print("❌ Database connection failed:", e)
        raise

def init_db():
    """Initialize essential database tables if they don't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Create login_logs table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS login_logs (
                log_id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Create an index for faster analytics
        cur.execute("CREATE INDEX IF NOT EXISTS idx_login_logs_user_id ON login_logs(user_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_login_logs_time ON login_logs(login_time);")
        
        conn.commit()
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
