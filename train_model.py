import cv2
import os
import numpy as np

data_dir = "data"
people = os.listdir(data_dir)

faces = []
labels = []
label_map = {}   # int → name
current_label = 0

for person in people:
    person_dir = os.path.join(data_dir, person)
    if not os.path.isdir(person_dir):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

if len(faces) == 0:
    print("No face images found. Run capture_faces.py first.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)
recognizer.save("face_model.yml")

# save label map
import json
with open("label_map.json", "w") as f:
    json.dump(label_map, f)

print("Model trained and saved as face_model.yml")
print("Label map:", label_map)
