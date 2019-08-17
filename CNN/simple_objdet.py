# OpenCV practice

import numpy as np

import cv2

''' Simple Object-detection using color '''

# color intensity
def getHSV(color):
    color = np.uint8([[color]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h = hsv_color[0][0][0]

    low = np.array([h - 10, 100, 100])
    high = np.array([h + 10, 255, 255])

    return low, high


low_red, high_red = getHSV([0, 0, 255])

low_green, high_green = getHSV([0, 255, 0])

low_blue, high_blue = getHSV([255, 0, 0])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap.open()

# read frames and detect object
while True:
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, low_red, high_red)
    green_mask = cv2.inRange(hsv, low_green, high_green)
    blue_mask = cv2.inRange(hsv, low_blue, high_blue)

    out_red = cv2.bitwise_and(frame, frame, mask=red_mask)
    out_green = cv2.bitwise_and(frame, frame, mask=green_mask)
    out_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    out = cv2.add(out_red, out_green)
    out = cv2.add(out, out_blue)

    cv2.imshow('original', frame)
    cv2.imshow('detect', out)
    
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

# finish
cap.release()
cv2.destroyAllWindows()
