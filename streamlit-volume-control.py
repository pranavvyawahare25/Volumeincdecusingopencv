import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = hands.process(imgRGB)
        
        # Lists for landmark positions
        lmList = []
        
        # If hands are detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                
                # Get all landmark positions
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
        
        # If landmarks are detected
        if len(lmList) != 0:
            # Get positions for thumb and index finger
            thumb_x, thumb_y = lmList[4][1], lmList[4][2]  # Thumb tip
            index_x, index_y = lmList[8][1], lmList[8][2]  # Index finger tip
            
            # Draw circles at the tips
            cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 3)
            
            # Calculate distance between fingers
            length = math.hypot(index_x - thumb_x, index_y - thumb_y)
            
            # Hand range: 50 - 300
            # Volume range: -65.25 - 0
            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volBar = np.interp(length, [50, 300], [400, 150])
            volPercentage = np.interp(length, [50, 300], [0, 100])
            
            # Reduce resolution to make it smoother
            vol = round(vol, 2)
            
            # Set system volume
            try:
                volume.SetMasterVolumeLevel(vol, None)
            except Exception as e:
                st.error(f"Error setting volume: {e}")
            
            # Draw volume bar
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPercentage)}%', (40, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Set page configuration
    st.set_page_config(page_title="Hand Gesture Volume Control", layout="wide")
    
    # Add title
    st.title("Hand Gesture Volume Control")
    
    # Add description
    st.markdown("""
    Control your system volume using hand gestures:
    - Show your hand to the camera
    - Use your thumb and index finger to control the volume
    - The distance between your fingers determines the volume level
    """)
    
    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="hand-gesture-volume",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
