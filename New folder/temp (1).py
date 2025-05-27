from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

frequency = 2500
duration = 1000

# Constants for thresholds and frame checks
earThresh = 0.3
earFrames = 48
marThresh = 0.5
marFrames = 15
orientationThresh = 20  # Threshold for face orientation (in degrees)

# Initialize Dlib's face detector and shape predictor
shapePredictor = r"C:\Users\sampa\Downloads\archive (1)\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

# Get indexes of facial landmarks for eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Function to calculate eye aspect ratio (EAR)
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate mouth aspect ratio (MAR)
def mouthAspectRatio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# Function to calculate face orientation using landmarks
def faceOrientation(shape):
    # Calculate angle between nose tip and chin (vector) vs. horizontal (x-axis)
    nose_tip = shape[30]
    chin = shape[8]
    dx = nose_tip[0] - chin[0]
    dy = nose_tip[1] - chin[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)

# Function to check face clarity
def faceClarity(shape, minLandmarkDistance=15):
    # Calculate distances between some key landmarks (nose tip to eyes, etc.)
    nose_tip = shape[30]
    left_eye = shape[42]
    right_eye = shape[39]
    
    nose_to_left_eye_dist = dist.euclidean(nose_tip, left_eye)
    nose_to_right_eye_dist = dist.euclidean(nose_tip, right_eye)
    
    # A rough threshold to decide if the face is clear
    if nose_to_left_eye_dist < minLandmarkDistance or nose_to_right_eye_dist < minLandmarkDistance:
        return False  # Face is not clear
    else:
        return True  # Face is clear

# Initialize counters
eyeCounter = 0
mouthCounter = 0

# Placeholder for ground truth and predictions
ground_truth = []  # Store the actual labels (Alert/Drowsy)
predictions = []   # Store the system's predictions (Alert/Drowsy)

# Lists to store EAR and MAR values for plotting
ear_values = []
mar_values = []

# Start video capture
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    # Check if no face is detected
    if len(rects) == 0:
        print("ALERT: No face detected!")
        cv2.putText(frame, "ALERT: NO FACE DETECTED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        winsound.Beep(frequency, duration)
    else:
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Calculate face orientation
            orientation = faceOrientation(shape)
            
            # Check if the orientation is below the threshold (face clear and facing camera)
            if orientation < orientationThresh:
                print("ALERT: Face not clear! Driver's face turned to side or not facing camera.")
                cv2.putText(frame, "ALERT: FACE TURNED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(frequency, duration)
                continue

            # Check for face clarity
            if not faceClarity(shape):
                print("ALERT: Face not clear!")
                cv2.putText(frame, "ALERT: FACE NOT CLEAR!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(frequency, duration)
                continue

            # Extract eye regions and compute EAR
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eyeAspectRatio(leftEye)
            rightEAR = eyeAspectRatio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            ear_values.append(ear)  # Store EAR value

            # Extract mouth region and compute MAR
            mouth = shape[mStart:mEnd]
            mar = mouthAspectRatio(mouth)
            mar_values.append(mar)  # Store MAR value

            # Console output for EAR and MAR values
            print(f"EAR: {ear:.2f}, MAR: {mar:.2f}")

            # Draw contours around the eyes and mouth
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # Check for eye closure (drowsiness detection)
            if ear < earThresh:
                eyeCounter += 1
                if eyeCounter >= earFrames:
                    print("DROWSINESS ALERT: Eyes closed for prolonged duration!")
                    cv2.putText(frame, "DROWSINESS DETECTED (EYES)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    winsound.Beep(frequency, duration)
                    predictions.append("Drowsy")
                    ground_truth.append("Drowsy")  # You can change this depending on the actual label
            else:
                eyeCounter = 0
                predictions.append("Alert")
                ground_truth.append("Alert")  # You can change this depending on the actual label

            # Check for yawning (mouth open)
            if mar > marThresh:
                mouthCounter += 1
                if mouthCounter >= marFrames:
                    print("DROWSINESS ALERT: Yawning detected!")
                    cv2.putText(frame, "DROWSINESS DETECTED (YAWNING)", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    winsound.Beep(frequency, duration)
                    predictions.append("Drowsy")
                    ground_truth.append("Drowsy")  # You can change this depending on the actual label
            else:
                mouthCounter = 0
                predictions.append("Alert")
                ground_truth.append("Alert")  # You can change this depending on the actual label

    # Display the frame with contours and alerts
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Release the camera and close windows
cam.release()
cv2.destroyAllWindows()

# Check if there are predictions made
if len(predictions) > 0 and len(ground_truth) > 0:
    # Calculate accuracy
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Classification report for precision, recall, F1-score
    report = classification_report(ground_truth, predictions, labels=["Alert", "Drowsy"], zero_division=0)
    print("Classification Report:")
    print(report)

    # Plot EAR and MAR values over time
    plt.figure(figsize=(10, 6))
    plt.plot(ear_values, label="EAR (Eye Aspect Ratio)", color='blue')
    plt.plot(mar_values, label="MAR (Mouth Aspect Ratio)", color='red')
    plt.axhline(y=earThresh, color='blue', linestyle='--', label="EAR Threshold")
    plt.axhline(y=marThresh, color='red', linestyle='--', label="MAR Threshold")
    plt.title("EAR and MAR Values Over Time")
    plt.xlabel("Frames")
    plt.ylabel("Ratio")

    plt.legend()
    plt.grid()
    plt.show()