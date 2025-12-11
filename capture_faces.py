import cv2
import os

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

name = input("Enter person name: ").strip()
save_dir = os.path.join("data", name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
max_images = 50  # images per person

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        file_path = os.path.join(save_dir, f"{name}_{count}.jpg")
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    cv2.putText(frame, f"Images: {count}/{max_images}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print("Face data collection completed.")
