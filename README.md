# Drowsiness Detection System

This is a web-based drowsiness detection system that uses computer vision to monitor for signs of drowsiness.

## Setup Instructions

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the shape predictor file:
   - Download the file "shape_predictor_68_face_landmarks.dat" from:
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract the .dat file and rename it to "shape_predictor_68_face_landmarks (1).dat"
   - Place it in the same directory as app.py

3. Run the application:
   ```
   python app.py
   ```

4. Access the application:
   - Open a web browser and go to http://localhost:5000
   - Login with username: admin, password: admin123

## Features
- Real-time drowsiness detection
- Video recording capability
- Web interface with login system
- Alerts for drowsiness detection 