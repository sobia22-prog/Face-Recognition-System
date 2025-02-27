import sqlite3
import face_recognition
import numpy as np

db_path = "face_recognition.db"
conn = sqlite3.connect(db_path)

def calculate_combined_encoding(image_paths):
    encodings = []
    for image_path in image_paths:
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                encodings.append(face_encodings[0])
            else:
                print(f"No face detected in image: {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    if not encodings:
        return None

    combined_encoding = np.mean(encodings, axis=0)
    return combined_encoding

def add_person(name, age, gender, image1_path, image2_path, image3_path, image4_path):
    image_paths = [image1_path, image2_path, image3_path, image4_path]

    combined_encoding = calculate_combined_encoding(image_paths)

    if combined_encoding is None:
        print(f"Could not calculate encoding for {name}. Skipping...")
        return

    try:
        conn.execute('''
        INSERT INTO People (name, age, gender, image1_path, image2_path, image3_path, image4_path, face_encoding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, age, gender, image1_path, image2_path, image3_path, image4_path, combined_encoding.tobytes()))

        conn.commit()
        print(f"Successfully added {name} to the database.")
    except Exception as e:
        print(f"Error adding {name}: {e}")

add_person(
    "Shane Dewson", "30", "Male",
    "database/ShaneDewson.jpeg",
    "database/ShaneDewson1.jpeg",
    "database/ShaneDewson2.jpeg",
    "database/ShaneDewson3.jpeg"
)

conn.close()
