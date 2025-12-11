import cv2
import json
import sqlite3
from datetime import datetime

# Load model & labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("label_map.json", "r") as f:
    label_map = json.load(f)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# DB connection
conn = sqlite3.connect("attendance.db")
c = conn.cursor()

# to avoid duplicate entries in one session
marked = set()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)
        name = label_map.get(str(label), label_map.get(label, "Unknown"))

        # mark attendance if confident enough
        if confidence < 70:  # lower = better
            cv2.putText(frame, f"{name} ({int(confidence)})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            if name not in marked:
                c.execute("INSERT INTO attendance (name) VALUES (?)", (name,))
                conn.commit()
                marked.add(name)
        else:
            cv2.putText(frame, "Unknown",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
print("Session ended. Attendance saved in attendance.db")
