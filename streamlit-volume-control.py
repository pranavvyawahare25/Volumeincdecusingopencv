import streamlit as st
import cv2
import numpy as np
import time
import os

# EyeBlinkDetector class
class EyeBlinkDetector:
    def __init__(self):
        # Get the path to the haar cascade files
        cascade_path = cv2.data.haarcascades
        face_cascade_path = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        eye_cascade_path = os.path.join(cascade_path, 'haarcascade_eye.xml')
        
        # Verify cascade files exist
        if not os.path.exists(face_cascade_path):
            raise FileNotFoundError(f"Face cascade file not found at: {face_cascade_path}")
        if not os.path.exists(eye_cascade_path):
            raise FileNotFoundError(f"Eye cascade file not found at: {eye_cascade_path}")
            
        # Load the pre-trained haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier()
        self.eye_cascade = cv2.CascadeClassifier()
        
        # Load the cascades and check if loaded successfully
        if not self.face_cascade.load(face_cascade_path):
            raise RuntimeError("Error loading face cascade classifier")
        if not self.eye_cascade.load(eye_cascade_path):
            raise RuntimeError("Error loading eye cascade classifier")
        
        print("Cascade classifiers loaded successfully")
        
        # Initialize parameters
        self.blink_counter = 0
        self.total_blinks = 0
        self.frame_count = 0
        self.last_eyes_detected = True
        self.start_time = time.time()
        
        # Parameters for improved blink detection
        self.no_eye_frames = 0
        self.MIN_FRAMES_EYES_CLOSED = 2  # Minimum frames eyes must be closed to count as blink
        self.MAX_FRAMES_EYES_CLOSED = 6  # Maximum frames eyes can be closed to count as blink
        
    def detect_blink(self, prev_eyes_detected, current_eyes_detected):
        """
        Improved blink detection with frame counting
        Returns True if a blink is detected
        """
        if not current_eyes_detected:
            self.no_eye_frames += 1
        elif current_eyes_detected:
            # If eyes were closed for an appropriate duration, count as blink
            if self.MIN_FRAMES_EYES_CLOSED <= self.no_eye_frames <= self.MAX_FRAMES_EYES_CLOSED:
                self.total_blinks += 1
                self.no_eye_frames = 0
                return True
            self.no_eye_frames = 0
        return False
            
    def calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.start_time)
        self.start_time = current_time
        return fps

    def process_frame(self, frame):
        """Process a single frame for eye detection"""
        if frame is None:
            raise ValueError("Invalid frame: None")
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        eyes_detected = False
        
        # Process each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # Draw rectangles around eyes and update detection flag
            if len(eyes) >= 2:  # Both eyes detected
                eyes_detected = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Detect blink
        blink_detected = self.detect_blink(self.last_eyes_detected, eyes_detected)
        self.last_eyes_detected = eyes_detected
        
        # Calculate FPS
        fps = self.calculate_fps()
        
        # Add text to frame
        cv2.putText(frame, f'Blinks: {self.total_blinks}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add eye state indicator
        eye_state = "Eyes Open" if eyes_detected else "Eyes Closed"
        cv2.putText(frame, eye_state, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if eyes_detected else (0, 0, 255), 2)
        
        # Return the processed frame as a bytes stream
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

# Create the detector instance
detector = EyeBlinkDetector()

# Streamlit App
st.title("Eye Blink Detection")

# Live Video Feed
context = st.context
video_capture = cv2.VideoCapture(0)

run_app = st.checkbox("Run App")
while run_app:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process the frame
    processed_frame_bytes = detector.process_frame(frame)

    # Display the processed frame
    st.image(processed_frame_bytes, width=640)

    # Break loop on 'Stop' button press (simulated by unchecking the checkbox)

    # Cleanup
video_capture.release()
cv2.destroyAllWindows()

