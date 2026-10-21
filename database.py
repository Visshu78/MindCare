import sqlite3
import os

# Resolve DB path — defaults to journal.db inside MindCare_v2 directory
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "journal.db")
)


def get_conn():
    """Return a live SQLite connection using the configured DB path."""
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         email TEXT UNIQUE,
         username TEXT UNIQUE,
         password TEXT)
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS entries
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_id INTEGER,
         date TEXT,
         entry_text TEXT,
         emotion TEXT,
         sentiment TEXT,
         confidence REAL,
         FOREIGN KEY (user_id) REFERENCES users (id))
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_settings
        (user_id INTEGER PRIMARY KEY,
         daily_reminder BOOLEAN,
         reminder_time TEXT,
         FOREIGN KEY (user_id) REFERENCES users (id))
    ''')

    # Automatic schema migration for existing databases
    c.execute("PRAGMA table_info(users)")
    user_cols = [col[1] for col in c.fetchall()]
    if "email" not in user_cols:
        c.execute("ALTER TABLE users ADD COLUMN email TEXT")

    c.execute("PRAGMA table_info(entries)")
    entry_cols = [col[1] for col in c.fetchall()]
    if "user_id" not in entry_cols:
        c.execute("ALTER TABLE entries ADD COLUMN user_id INTEGER")
    if "emotion" not in entry_cols:
        c.execute("ALTER TABLE entries ADD COLUMN emotion TEXT")
    if "sentiment" not in entry_cols:
        c.execute("ALTER TABLE entries ADD COLUMN sentiment TEXT")
    if "confidence" not in entry_cols:
        c.execute("ALTER TABLE entries ADD COLUMN confidence REAL")

    conn.commit()
    conn.close()


def get_user_entries(user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute('''SELECT date, entry_text, emotion, sentiment, confidence 
                 FROM entries WHERE user_id = ? ORDER BY date DESC''', (user_id,))
    entries = c.fetchall()
    conn.close()
    return entries


def add_user_entry(user_id, date, entry_text, emotion, sentiment, confidence):
    conn = get_conn()
    c = conn.cursor()
    c.execute('''INSERT INTO entries (user_id, date, entry_text, emotion, sentiment, confidence)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (user_id, date, entry_text, emotion, sentiment, confidence))
    conn.commit()
    conn.close()