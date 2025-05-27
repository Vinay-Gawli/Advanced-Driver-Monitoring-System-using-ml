from flask import Flask, jsonify, Response, render_template, request, send_from_directory, redirect
import threading
import cv2
import time
import numpy as np
import winsound
import os
from datetime import datetime
from queue import Queue
import imutils

app = Flask(__name__)
detection_running = False
status_message = "Idle"

# Global variables for detection metrics
ear_value = 0.0
mar_value = 0.0
is_alert = False
alert_message = ""

# Constants for thresholds
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Detection thresholds
EYE_CLOSED_THRESHOLD = 30  # frames
YAWN_THRESHOLD = 0.5  # Lowered threshold for easier yawn detection
YAWN_FRAMES_THRESHOLD = 5  # Reduced frames needed for yawn detection
HEAD_TILT_THRESHOLD = 0.15
HEAD_TILT_FRAMES = 3

# Counters and state tracking
eye_closed_counter = 0
yawn_counter = 0
head_tilt_counter = 0
last_alert_time = 0
ALERT_COOLDOWN = 2.0  # Reduced cooldown between alerts

# Sound alert settings
DROWSY_BEEP_FREQ = 1000  # Hz (lowered frequency)
YAWN_BEEP_FREQ = 1500   # Hz
HEAD_BEEP_FREQ = 2000   # Hz
BEEP_DURATION = 200     # ms (shortened duration)

# Global variables for video recording
recording = False
output_video = None
recording_start_time = None
recorded_videos = []

# Global camera instance and frame queue
camera = None
frame_queue = Queue(maxsize=2)
last_frame = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    try:
        # Release any existing camera
        if camera is not None:
            camera.release()
            cv2.destroyAllWindows()
        camera = None
        
        print("\nInitializing camera...")
        
        # Try DirectShow first (most stable on Windows)
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if camera.isOpened():
            # Basic configuration - keep it minimal for stability
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Wait for camera to initialize
            time.sleep(1)
            
            # Test camera stability
            success_count = 0
            for _ in range(5):
                ret, frame = camera.read()
                if ret and frame is not None and frame.size > 0:
                    success_count += 1
                time.sleep(0.1)
            
            if success_count >= 3:
                print("Camera initialized successfully!")
                return camera
            
            print("Camera not reading frames reliably")
            camera.release()
            
        # If DirectShow failed, try default backend
        print("Trying default backend...")
        camera = cv2.VideoCapture(0)
        
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(1)
            
            success_count = 0
            for _ in range(5):
                ret, frame = camera.read()
                if ret and frame is not None and frame.size > 0:
                    success_count += 1
                time.sleep(0.1)
            
            if success_count >= 3:
                print("Camera initialized successfully with default backend!")
                return camera
        
        print("Failed to initialize camera with any backend")
        return None
            
    except Exception as e:
        print(f"Error in camera initialization: {str(e)}")
        if camera is not None:
            camera.release()
        return None

def play_alert(frequency, duration):
    """Play alert sound in a separate thread"""
    try:
        winsound.Beep(frequency, duration)
    except Exception as e:
        print(f"Sound alert error: {e}")

def process_frame_for_detection(frame):
    if frame is None:
        return None

    try:
        global eye_closed_counter, yawn_counter, head_tilt_counter, last_alert_time
        global status_message, ear_value, mar_value, is_alert, alert_message
        
        current_time = time.time()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            maxSize=(400, 400)
        )

        if len(faces) == 0:
            if current_time - last_alert_time >= ALERT_COOLDOWN:
                status_message = "No face detected"
                is_alert = True
                alert_message = "⚠️ No face detected!"
                play_alert(HEAD_BEEP_FREQ, BEEP_DURATION)
                last_alert_time = current_time
            return frame

        # Process the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Head position detection
        face_center_x = x + w//2
        frame_center_x = frame.shape[1]//2
        deviation = abs(face_center_x - frame_center_x) / frame.shape[1]
        
        # Visual indicators for head position
        cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame.shape[0]), (0, 255, 0), 1)
        cv2.line(frame, (face_center_x, y), (face_center_x, y+h), (0, 0, 255), 2)
        
        if deviation > HEAD_TILT_THRESHOLD:
            head_tilt_counter += 1
            if head_tilt_counter >= HEAD_TILT_FRAMES and current_time - last_alert_time >= ALERT_COOLDOWN:
                cv2.putText(frame, "LOOK AT THE ROAD", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                status_message = "Look at the road!"
                is_alert = True
                alert_message = "👀 LOOK AT THE ROAD"
                play_alert(HEAD_BEEP_FREQ, BEEP_DURATION)
                last_alert_time = current_time
        else:
            head_tilt_counter = max(0, head_tilt_counter - 1)

        # Improved yawn detection
        mouths = mouth_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=11,  # Reduced for better detection
            minSize=(int(w*0.4), int(h*0.1)),  # Adjusted relative to face size
            maxSize=(int(w*0.7), int(h*0.4))
        )

        if len(mouths) > 0:
            mouth = max(mouths, key=lambda m: m[2] * m[3])
            (mx, my, mw, mh) = mouth
            mar = float(mh) / mw
            mar_value = mar
            
            # Draw mouth rectangle with color feedback
            color = (0, 255, 0) if mar <= YAWN_THRESHOLD else (0, 0, 255)
            cv2.rectangle(face_roi_color, (mx, my), (mx+mw, my+mh), color, 2)
            
            # Show MAR value
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if mar > YAWN_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= YAWN_FRAMES_THRESHOLD and current_time - last_alert_time >= ALERT_COOLDOWN:
                    cv2.putText(frame, "YAWNING DETECTED!", (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    status_message = "Yawning detected"
                    is_alert = True
                    alert_message = "🥱 YAWNING DETECTED"
                    play_alert(YAWN_BEEP_FREQ, BEEP_DURATION)
                    last_alert_time = current_time
            else:
                yawn_counter = max(0, yawn_counter - 1)
        else:
            yawn_counter = max(0, yawn_counter - 1)
            mar_value = 0.0

        # Add detection metrics overlay
        metrics = [
            f"Head Deviation: {deviation:.2f}",
            f"MAR: {mar_value:.2f}",
            f"Yawn Counter: {yawn_counter}/{YAWN_FRAMES_THRESHOLD}",
            f"Head Tilt: {head_tilt_counter}/{HEAD_TILT_FRAMES}",
            f"Status: {status_message}"
        ]
        
        y_pos = frame.shape[0] - 150
        for metric in metrics:
            cv2.putText(frame, metric, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 30

        return frame

    except Exception as e:
        print(f"Error in process_frame_for_detection: {e}")
        return frame

def video_capture_loop():
    global last_frame, camera
    frame_count = 0
    retry_count = 0
    max_retries = 3
    last_successful_read = time.time()
    
    while True:
        try:
            current_time = time.time()
            
            # Check if camera needs initialization
            if camera is None or not camera.isOpened():
                if retry_count < max_retries:
                    print(f"\nAttempting to initialize camera (attempt {retry_count + 1}/{max_retries})")
                    camera = get_camera()
                    retry_count += 1
                    if camera is None:
                        time.sleep(2)
                        continue
                else:
                    print("Max retries reached. Waiting before trying again...")
                    time.sleep(5)
                    retry_count = 0
                    continue
            
            # Read frame with timeout
            ret, frame = camera.read()
            
            if not ret or frame is None or frame.size == 0:
                # Only reinitialize if we haven't gotten a frame for 3 seconds
                if current_time - last_successful_read > 3:
                    print("No valid frames for 3 seconds, reinitializing camera...")
                    if camera is not None:
                        camera.release()
                    camera = None
                continue
            
            # Update successful read time
            last_successful_read = current_time
            
            # Process frame if detection is running
            if detection_running:
                try:
                    processed_frame = process_frame_for_detection(frame.copy())
                    if processed_frame is not None:
                        frame = processed_frame
                except Exception as e:
                    print(f"Error in detection processing: {e}")
            
            # Add frame information
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update last frame
            last_frame = frame.copy()
            frame_count += 1
            
            # Reset retry count on successful frames
            if frame_count % 30 == 0:
                retry_count = 0
            
            # Small delay to prevent busy waiting
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in video capture loop: {e}")
            if camera is not None:
                camera.release()
            camera = None
            time.sleep(1)
            retry_count += 1

# Start the video capture thread
capture_thread = threading.Thread(target=video_capture_loop)
capture_thread.daemon = True
capture_thread.start()

@app.route('/status')
def get_status():
    global status_message, ear_value, mar_value, is_alert, alert_message
    return jsonify({
        'status': status_message,
        'ear': ear_value,
        'mar': mar_value,
        'isAlert': is_alert,
        'alertMessage': alert_message
    })

@app.route('/video_feed')
def video_feed():
    def generate():
        frame_error_count = 0
        max_errors = 3
        
        while True:
            try:
                if last_frame is None:
                    frame_error_count += 1
                    if frame_error_count > max_errors:
                        # Create an error frame with troubleshooting info
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "Camera Not Available", (50, 200),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(frame, "Please check:", (50, 240),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, "1. Camera connection", (70, 280),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, "2. Camera permissions", (70, 320),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, "3. Other applications using camera", (70, 360),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    frame = last_frame.copy()
                    frame_error_count = 0

                # Ensure the frame is in BGR format
                if len(frame.shape) == 2:  # If grayscale, convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Add timestamp to verify frame is updating
                cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), 
                          (frame.shape[1] - 150, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, 
                          (0, 255, 0), 
                          2)

                # Convert to JPEG with good quality
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print(f"Error in video feed generate: {str(e)}")
                time.sleep(0.1)
                continue

            # Maintain approximately 30 FPS
            time.sleep(0.033)

    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_running, status_message
    detection_running = True
    status_message = "Active"
    return jsonify({'status': 'started'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_running, status_message
    detection_running = False
    status_message = "Stopped"
    return jsonify({'status': 'stopped'})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, output_video, recording_start_time
    
    if not recording:
        if not os.path.exists('recordings'):
            os.makedirs('recordings')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/drowsiness_recording_{timestamp}.avi"
        
        if last_frame is not None:
            height, width = last_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_video = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            recording = True
            recording_start_time = datetime.now()
            return jsonify({'status': 'recording_started', 'filename': filename})
        
        return jsonify({'status': 'error', 'message': 'No video frame available'})
    
    return jsonify({'status': 'already_recording'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, output_video, recording_start_time, recorded_videos
    
    if recording and output_video is not None:
        recording = False
        output_video.release()
        
        duration = (datetime.now() - recording_start_time).total_seconds()
        
        recorded_videos.append({
            'filename': output_video.filename,
            'timestamp': recording_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'duration': f"{int(duration // 60)}:{int(duration % 60):02d}"
        })
        
        output_video = None
        return jsonify({'status': 'recording_stopped', 'duration': duration})
    
    return jsonify({'status': 'not_recording'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        if username == 'admin' and password == 'admin123':
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'})
    return send_from_directory('.', 'login.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

@app.route('/')
def root():
    return redirect('/login')

@app.route('/logout', methods=['POST'])
def logout():
    global detection_running, recording, output_video
    try:
        detection_running = False
        if recording and output_video is not None:
            recording = False
            output_video.release()
            output_video = None
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error during logout: {e}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 