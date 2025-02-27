import sqlite3

conn = sqlite3.connect('face_recognition.db')

conn.execute('''
CREATE TABLE IF NOT EXISTS People (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age TEXT NOT NULL,
    gender TEXT NOT NULL,
    image1_path TEXT NOT NULL,
    image2_path TEXT NOT NULL,
    image3_path TEXT NOT NULL,
    image4_path TEXT NOT NULL,
    face_encoding BLOB NOT NULL
);
''')

print("Database and table created successfully.")

conn.close()
