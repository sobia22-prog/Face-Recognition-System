import cv2
import sqlite3
import face_recognition
import numpy as np
import os
import tensorflow as tf
from datetime import datetime, timedelta
import argparse

def load_encodings_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age, gender, face_encoding FROM People")
    rows = cursor.fetchall()
    conn.close()

    known_encodings = []
    known_details = []

    for row in rows:
        person_id, name, age, gender, face_encoding = row
        encoding = np.frombuffer(face_encoding, dtype=np.float64)
        known_encodings.append(encoding)
        known_details.append({"id": person_id, "name": name, "age": age, "gender": gender})

    return known_encodings, known_details

def save_frame_with_labels(frame, labels, folder):
    os.makedirs(folder, exist_ok=True) 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")  
    filename = f"{folder}/frame_{timestamp}.jpg"

    for label, position in labels:
        cv2.putText(
            frame,
            label,
            (position[0], position[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (position[0], position[1]), (position[2], position[3]), (0, 255, 0), 2)

    cv2.imwrite(filename, frame)
    print(f"Frame saved to {filename}")

def process_frame(frame, known_encodings, known_details, last_capture_time, padding=20):
    current_time = datetime.now()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    labels = []
    has_unknown = False

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        top, right, bottom, left = face_location

        if True in matches:
            best_match_index = np.argmin(face_distances)
            matched_person = known_details[best_match_index]
            name = matched_person["name"]
            age = matched_person["age"]
            gender = matched_person["gender"]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}, {age}, {gender}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if matched_person["id"] not in last_capture_time or \
                    current_time - last_capture_time[matched_person["id"]] > timedelta(seconds=30):
                last_capture_time[matched_person["id"]] = current_time
                labels.append((f"{name}, {age}, {gender}", (left, top, right, bottom)))
        else:
            has_unknown = True

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            labels.append(("Unknown", (left, top, right, bottom)))

    if has_unknown:
        save_frame_with_labels(frame.copy(), labels, "unknown_faces")
    elif labels:
        save_frame_with_labels(frame.copy(), labels, "known_faces")

    return last_capture_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to the input image")
    args = parser.parse_args()

    db_path = "face_recognition.db"
    known_encodings, known_details = load_encodings_from_db(db_path)
    last_capture_time = {}

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Unable to load image {args.image}")
            return

        labels = process_frame(frame, known_encodings, known_details, last_capture_time)

        cv2.imshow("Face Recognition - Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Error: Unable to access the camera")
            return

        while True:
            ret, frame = video.read()
            if not ret:
                print("Error: Unable to retrieve frame from camera")
                break

            last_capture_time = process_frame(frame, known_encodings, known_details, last_capture_time)

            cv2.imshow("Face Recognition - Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break

        video.release()
        cv2.destroyAllWindows()

main()
