# Face Attendance System using OpenCV

A real-time face recognition–based attendance system built using OpenCV and SQLite.

## 🚀 Features
- Face image collection using webcam
- Face model training using LBPH algorithm
- Real-time face recognition
- Automatic attendance logging into SQLite database
- Timestamp-based entry storage

## 🛠️ Tech Stack
- Python  
- OpenCV (opencv-contrib-python)  
- NumPy  
- SQLite  

## ▶️ How to Run

1. Install dependencies:
   
pip install -r requirements.txt

2.Collect face images:

python capture_faces.py


3.Train the face recognition model:

python train_model.py


4.Initialize the attendance database:

python init_db.py


5.Start the face attendance system:

python attendance_camera.py

🔒 Data Privacy Notice

For privacy and security reasons, face image datasets and attendance databases are not included in this repository.

Users must collect their own face data using capture_faces.py. The required folder structure will be automatically created when the script is executed.

Example structure after data collection:

data/
 ├── Person1/
 ├── Person2/
