import sqlite3

def init_db():
    conn = sqlite3.connect('journal.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn.commit()
    conn.close()

def get_user_entries(user_id):
    conn = sqlite3.connect('journal.db')
    c = conn.cursor()
    c.execute('''SELECT date, entry_text, emotion, sentiment, confidence 
                 FROM entries WHERE user_id = ? ORDER BY date DESC''', (user_id,))
    entries = c.fetchall()
    conn.close()
    return entries

def add_user_entry(user_id, date, entry_text, emotion, sentiment, confidence):
    conn = sqlite3.connect('journal.db')
    c = conn.cursor()
    c.execute('''INSERT INTO entries (user_id, date, entry_text, emotion, sentiment, confidence)
                 VALUES (?, ?, ?, ?, ?, ?)''', 
             (user_id, date, entry_text, emotion, sentiment, confidence))
    conn.commit()
    conn.close()