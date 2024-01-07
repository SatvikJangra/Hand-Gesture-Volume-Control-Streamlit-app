import cv2
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import mediapipe as mp
import streamlit as st

class HandDetector():
    def __init__(self, mode=False, max_hands=2, model_c=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_c = model_c
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_c, self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.pos_list = []  # Added this line

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_land in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_land, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_no=0, draw=True):
        x_list = []
        y_list = []
        bbox = []
        self.pos_list = []  # Added this line
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.pos_list.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.pos_list, bbox

    def fingers_up(self):
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if self.pos_list[tip_ids[0]][1] > self.pos_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.pos_list[tip_ids[id]][2] < self.pos_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True):
        x1, y1 = self.pos_list[p1][1], self.pos_list[p1][2]
        x2, y2 = self.pos_list[p2][1], self.pos_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = np.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    st.title("Hand Gesture Controlled Volume and Music Player")

    cap = cv2.VideoCapture(0)
    w_cam, h_cam = 640, 480
    cap.set(3, w_cam)
    cap.set(4, h_cam)

    p_time = 0
    detector = HandDetector()

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    vol_range = volume.GetVolumeRange()
    min_vol = vol_range[0]
    max_vol = vol_range[1]
    vol = 0
    vol_bar = 400
    vol_per = 0
    area = 0
    color_vol = (255, 0, 0)

    # Add Streamlit file uploader for selecting songs
    uploaded_file = st.file_uploader("Choose a song", type=["mp3", "wav"])

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/ogg')

    # Streamlit's magic command to display the OpenCV video stream
    video_stream = st.image([])

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        pos_list, bbox = detector.find_position(img)

        if len(pos_list) != 0:
            # Filter based on size
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
            if 500 <= area <= 1000:
                pass

            # Find Distance between index and Thumb
            length, img, line_info = detector.find_distance(4, 8, img)

            # Convert Volume
            vol_bar = np.interp(length, [50, 200], [400, 150])
            vol_per = np.interp(length, [50, 200], [0, 100])

            # We can Reduce Resolution to make it Smoother
            smoothness = 10
            vol_per = smoothness * round(vol_per / smoothness)

            # Check fingers up
            fingers = detector.fingers_up()

            # Check if pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(vol_per / 100, None)
                color_vol = (0, 255, 0)

        # Drawings
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(vol_per)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        c_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
        cv2.putText(img, f'Vol Set: {int(c_vol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color_vol, 3)

        # Frame rate
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # Display the updated frame in Streamlit
        video_stream.image(img, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
