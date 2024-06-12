import sqlite3

try:
    conn = sqlite3.connect('customer_faces_data.db')
    c = conn.cursor()
    c.execute("ALTER TABLE customers ADD COLUMN confirmed INTEGER DEFAULT 0;")
    conn.commit()
    print("Column added successfully.")
except sqlite3.OperationalError as e:
    print("SQLite error:", e)
finally:
    conn.close()
