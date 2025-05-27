y6from flask import Flask, jsonify, Response, render_template, request, send_from_directory, redirect, session, url_for
import threading
import cv2
import time
import numpy as np
import winsound
import os
from datetime import datetime
from queue import Queue
import imutils
import dlib
from scipy.spatial import distance as dist

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key_here'  # Required for session management
detection_running = False
status_message = "Idle"

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for facial landmarks
LEFT_EYE_START = 36
LEFT_EYE_END = 42
RIGHT_EYE_START = 42
RIGHT_EYE_END = 48
MOUTH_START = 48
MOUTH_END = 68

# Detection thresholds
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
MAR_THRESHOLD = 0.6   # Mouth Aspect Ratio threshold
HEAD_TILT_THRESHOLD = 0.2  # Head tilt threshold
CONSECUTIVE_FRAMES = 20  # Number of consecutive frames for alert

# Alert sounds
DROWSY_BEEP_FREQ = 2000  # Hz
YAWN_BEEP_FREQ = 1500   # Hz
HEAD_BEEP_FREQ = 1000   # Hz
BEEP_DURATION = 1000    # ms

# Counters and state variables
eye_closed_counter = 0
yawn_counter = 0
head_tilt_counter = 0
last_alert_time = 0
alert_cooldown = 3.0  # seconds between alerts

# Global variables for detection metrics
ear_value = 0.0
mar_value = 0.0
is_alert = False
alert_message = ""

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
camera_index = 0  # Current camera index

# Add yawn counter as a global variable at the top of the file with other globals
yawn_counter = 0

def get_camera():
    global camera
    try:
        # Release any existing camera
        if camera is not None:
            camera.release()
        camera = None
        
        print("\nTrying different camera backends...")
        
        # Try DirectShow first as it's most reliable on Windows
        camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        
        if not camera.isOpened():
            print("DirectShow failed, trying default...")
            camera = cv2.VideoCapture(0)
        
        if camera.isOpened():
            # Set camera properties - minimal settings to ensure it works
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Wait a bit for camera to initialize
            time.sleep(2)
            
            # Test read multiple frames
            for _ in range(5):  # Read 5 frames to warm up the camera
                ret, frame = camera.read()
                if ret and frame is not None and frame.size > 0:
                    print("Camera initialized successfully!")
                    print(f"Frame shape: {frame.shape}")
                    print(f"Frame mean brightness: {frame.mean():.2f}")
                    return camera
            
            print("Failed to get valid frames")
            camera.release()
            return None
        
        print("Failed to open camera")
        return None
            
    except Exception as e:
        print(f"Error initializing camera: {e}")
        if camera is not None:
            camera.release()
        return None

def release_camera():
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
    except Exception as e:
        print(f"Error releasing camera: {e}")
        camera = None

def calculate_ear(eye_points):
    # Calculate the vertical distances
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Calculate the horizontal distance
    C = dist.euclidean(eye_points[0], eye_points[3])
    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth_points):
    # Calculate the vertical distances
    A = dist.euclidean(mouth_points[2], mouth_points[10])  # Upper lip to lower lip
    B = dist.euclidean(mouth_points[4], mouth_points[8])   # Middle points
    # Calculate the horizontal distance
    C = dist.euclidean(mouth_points[0], mouth_points[6])   # Mouth width
    # Calculate the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar

def process_frame_for_detection(frame):
    if frame is None:
        return None

    try:
        global eye_closed_counter, yawn_counter, head_tilt_counter, last_alert_time
        global status_message, ear_value, mar_value, is_alert, alert_message
        
        current_time = time.time()
        frame_with_alerts = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using dlib
        faces = detector(gray)
        
        if len(faces) == 0:
            cv2.putText(frame_with_alerts, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame_with_alerts

        for face in faces:
            # Get facial landmarks
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Draw face rectangle
            cv2.rectangle(frame_with_alerts, (face.left(), face.top()),
                        (face.right(), face.bottom()), (255, 0, 0), 2)
            
            # Get eye coordinates
            left_eye = shape[LEFT_EYE_START:LEFT_EYE_END]
            right_eye = shape[RIGHT_EYE_START:RIGHT_EYE_END]
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Draw eye contours
            cv2.drawContours(frame_with_alerts, [np.array(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame_with_alerts, [np.array(right_eye)], -1, (0, 255, 0), 1)
            
            # Check for eye closure
            if avg_ear < EAR_THRESHOLD:
                eye_closed_counter += 1
                if eye_closed_counter >= CONSECUTIVE_FRAMES:
                    alert = "👉 EYES CLOSED — DROWSINESS DETECTED — BUZZER ON"
                    cv2.putText(frame_with_alerts, alert, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if current_time - last_alert_time > alert_cooldown:
                        threading.Thread(target=lambda: winsound.Beep(DROWSY_BEEP_FREQ, BEEP_DURATION),
                                      daemon=True).start()
                        last_alert_time = current_time
            else:
                eye_closed_counter = max(0, eye_closed_counter - 1)
            
            # Get mouth coordinates and calculate MAR
            mouth = shape[MOUTH_START:MOUTH_END]
            mar = calculate_mar(mouth)
            
            # Draw mouth contour
            cv2.drawContours(frame_with_alerts, [np.array(mouth)], -1, (0, 255, 0), 1)
            
            # Check for yawning
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= CONSECUTIVE_FRAMES // 2:  # Less frames needed for yawn detection
                    alert = "👉 YAWNING DETECTED — BUZZER ON"
                    cv2.putText(frame_with_alerts, alert, (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if current_time - last_alert_time > alert_cooldown:
                        threading.Thread(target=lambda: winsound.Beep(YAWN_BEEP_FREQ, BEEP_DURATION),
                                      daemon=True).start()
                        last_alert_time = current_time
            else:
                yawn_counter = max(0, yawn_counter - 1)
            
            # Head pose estimation using facial landmarks
            nose_bridge = shape[27:31]  # Nose bridge points
            left_right_diff = abs(nose_bridge[-1][0] - nose_bridge[0][0])
            head_tilt = left_right_diff / (face.right() - face.left())
            
            if head_tilt > HEAD_TILT_THRESHOLD:
                head_tilt_counter += 1
                if head_tilt_counter >= CONSECUTIVE_FRAMES // 2:
                    alert = "👉 LOOK AT THE ROAD — BUZZER ON"
                    cv2.putText(frame_with_alerts, alert, (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if current_time - last_alert_time > alert_cooldown:
                        threading.Thread(target=lambda: winsound.Beep(HEAD_BEEP_FREQ, BEEP_DURATION),
                                      daemon=True).start()
                        last_alert_time = current_time
            else:
                head_tilt_counter = max(0, head_tilt_counter - 1)
            
            # Add debug info
            debug_info = [
                f"EAR: {avg_ear:.2f}/{EAR_THRESHOLD:.2f}",
                f"MAR: {mar:.2f}/{MAR_THRESHOLD:.2f}",
                f"Head Tilt: {head_tilt:.2f}/{HEAD_TILT_THRESHOLD:.2f}",
                f"Eye Counter: {eye_closed_counter}/{CONSECUTIVE_FRAMES}",
                f"Yawn Counter: {yawn_counter}/{CONSECUTIVE_FRAMES//2}"
            ]
            
            y_pos = frame_with_alerts.shape[0] - 140
            for info in debug_info:
                cv2.putText(frame_with_alerts, info, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20

        return frame_with_alerts

    except Exception as e:
        print(f"Error in process_frame_for_detection: {e}")
        return frame

def video_capture_loop():
    global last_frame, camera
    frame_count = 0
    start_time = time.time()
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            if camera is None or not camera.isOpened():
                if retry_count < max_retries:
                    print(f"\nAttempting to initialize camera (attempt {retry_count + 1}/{max_retries})")
                    camera = get_camera()
                    retry_count += 1
                    if camera is None:
                        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(error_frame, "Camera Not Available", (50, 240),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(error_frame, "Check camera connection", (50, 280),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        last_frame = error_frame
                        time.sleep(2)
                        continue
                else:
                    print("Max retries reached. Please check your camera connection.")
                    time.sleep(5)
                    retry_count = 0
                    continue

            ret, frame = camera.read()
            
            if not ret or frame is None or frame.size == 0:
                print("Failed to read frame")
                camera.release()
                camera = None
                continue

            # Create a copy for modifications
            display_frame = frame.copy()

            # Add debug info
            cv2.putText(display_frame, f"Frame #{frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_frame, f"Mean: {frame.mean():.1f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if detection_running:
                try:
                    processed_frame = process_frame_for_detection(frame)
                    if processed_frame is not None:
                        display_frame = processed_frame
                except Exception as e:
                    print(f"Error in detection processing: {e}")

            # Update the last frame
            last_frame = display_frame

            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                print(f"FPS: {fps:.2f}, Mean brightness: {frame.mean():.2f}")
                start_time = time.time()

            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)

        except Exception as e:
            print(f"Error in video capture loop: {e}")
            if camera is not None:
                camera.release()
            camera = None
            time.sleep(1)
            retry_count += 1

# Start the video capture thread
video_thread = threading.Thread(target=video_capture_loop, daemon=True)
video_thread.start()

@app.route('/status')
def get_status():
    global status_message, ear_value, mar_value, is_alert, alert_message, camera
    
    camera_status = {
        'connected': camera is not None and camera.isOpened(),
        'fps': 0,
        'resolution': '640x480'
    }
    
    if camera is not None and camera.isOpened():
        camera_status['fps'] = camera.get(cv2.CAP_PROP_FPS)
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_status['resolution'] = f"{width}x{height}"
    
    return jsonify({
        'status': status_message,
        'ear': ear_value,
        'mar': mar_value,
        'isAlert': is_alert,
        'alertMessage': alert_message,
        'camera': camera_status,
        'detection_active': detection_running,
        'recording_active': recording
    })

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                if last_frame is None:
                    # Create an error frame if no frame is available
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Frame Available", (50, 240),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
                    frame = last_frame.copy()

                # Ensure the frame is in BGR format
                if len(frame.shape) == 2:  # If grayscale, convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Convert to JPEG with high quality
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                print(f"Error in generate: {e}")
                time.sleep(1)
                continue

    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Add a route to check camera status
@app.route('/camera_status')
def camera_status():
    global camera
    status = {
        'camera_initialized': camera is not None,
        'camera_opened': camera is not None and camera.isOpened() if camera is not None else False,
        'last_frame_available': last_frame is not None,
        'last_frame_brightness': float(last_frame.mean()) if last_frame is not None else 0.0
    }
    return jsonify(status)

# Add a route to force camera reinitialization
@app.route('/reinit_camera', methods=['POST'])
def reinit_camera():
    global camera
    try:
        if camera is not None:
            camera.release()
        camera = get_camera()
        return jsonify({
            'success': camera is not None and camera.isOpened(),
            'message': 'Camera reinitialized successfully' if camera is not None else 'Failed to reinitialize camera'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error reinitializing camera: {str(e)}'
        })

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_running
    try:
        if not detection_running:
            detection_running = True
            return jsonify({'success': True, 'message': 'Detection started'})
        return jsonify({'success': True, 'message': 'Detection already running'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_running
    try:
        detection_running = False
        return jsonify({'success': True, 'message': 'Detection stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, output_video, recording_start_time
    
    try:
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
                return jsonify({
                    'success': True,
                    'message': 'Recording started',
                    'filename': filename
                })
            
            return jsonify({
                'success': False,
                'message': 'No video frame available'
            })
        
        return jsonify({
            'success': False,
            'message': 'Recording already in progress'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting recording: {str(e)}'
        })

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, output_video, recording_start_time, recorded_videos
    
    try:
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
            return jsonify({
                'success': True,
                'message': 'Recording stopped',
                'duration': duration
            })
        
        return jsonify({
            'success': False,
            'message': 'No active recording'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error stopping recording: {str(e)}'
        })

@app.route('/recordings')
def get_recordings():
    return jsonify({'recordings': recorded_videos})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('serve_dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/dashboard')
def serve_dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/')
def root():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return redirect(url_for('serve_dashboard'))

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)