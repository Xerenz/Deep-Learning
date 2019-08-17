# OpenCV practice

''' Image processing '''

import numpy as np
import cv2

''' Changing Colorspaces '''

# get list of flags
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

# learning how to capture video
cap = cv2.VideoCapture(0)

while True:
    # read frame-by-frame
    ret, frame = cap.read()

    # turn frame to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # show frames
    cv2.imshow('gray', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when finished release capture
cap.release()
cv2.destroyAllWindows()

''' Color detection '''

# color detection in captured frame
cap = cv2.VideoCapture(0)

# check
if not cap.isOpened():
    cap.open()

# set lower and upper values of the color
green = np.uint8([[[0, 255, 0]]])
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
H = hsv_green[0][0][0]

while True:
    _, frame = cap.read()

    # get hsv frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    
    # mask everything except stuff in range
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # narrow down detected object
    detect = cv2.bitwise_and(frame, frame, mask=mask)

    # display
    cv2.imshow('original', frame)
    cv2.imshow('detected', detect)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# finish
cap.release()
cv2.destroyAllWindows()

    

    

    




















