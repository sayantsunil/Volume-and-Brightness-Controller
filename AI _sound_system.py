import cv2
import mediapipe as mp
import numpy as np
import math
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# PyCaw setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume_ctrl.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam setup
cap = cv2.VideoCapture(0)

def fingers_up(lm_list):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(1 if lm_list[4].x < lm_list[3].x else 0)  # Thumb
    for tip_id in tips_ids[1:]:
        fingers.append(1 if lm_list[tip_id].y < lm_list[tip_id - 2].y else 0)
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm_list = hand_landmarks.landmark
        h, w, _ = img.shape

        x1, y1 = int(lm_list[4].x * w), int(lm_list[4].y * h)
        x2, y2 = int(lm_list[8].x * w), int(lm_list[8].y * h)

        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        fingers = fingers_up(lm_list)

        dist = math.hypot(x2 - x1, y2 - y1)

        if fingers == [1, 1, 0, 0, 0]:
            # Volume control
            vol = np.interp(dist, [20, 150], [min_vol, max_vol])
            vol_bar = np.interp(dist, [20, 150], [400, 150])
            vol_percent = np.interp(dist, [20, 150], [0, 100])

            volume_ctrl.SetMasterVolumeLevel(vol, None)

            # Draw volume bar
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(vol_percent)} %', (40, 430), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

        elif fingers == [1, 1, 1, 1, 1]:
            # Brightness control (smooth)
            try:
                brightness = np.interp(dist, [20, 150], [0, 100])
                brightness_bar = np.interp(dist, [20, 150], [400, 150])
                sbc.set_brightness(int(brightness), method='wmi')

                # Brightness bar
                cv2.rectangle(img, (550, 150), (585, 400), (255, 255, 0), 2)
                cv2.rectangle(img, (550, int(brightness_bar)), (585, 400), (255, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(brightness)} %', (530, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
            except Exception:
                cv2.putText(img, 'Brightness Error', (400, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        else:
            cv2.putText(img, 'Gesture: Volume or Brightness?', (30, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Gesture Volume & Brightness Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
