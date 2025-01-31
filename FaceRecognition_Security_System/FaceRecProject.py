import cv2
import numpy as np
import face_recognition
import os
import requests
from datetime import datetime
import time

# Motion detection parameters
motion_threshold = 200

# Face recognition parameters
path = "ImagesAttendance"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
classNames.append('Unknown')
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Setup HTTP endpoint for ESP32
esp32_ip = "http://192.168.100.132/status"  # ESP32 IP with endpoint "/status"

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize variables for motion detection and face detection
previous_frame = None
recognized_faces = {}

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Convert the frame to grayscale for motion detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Perform motion detection
    if previous_frame is None:
        previous_frame = gray
        continue

    frame_delta = cv2.absdiff(previous_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > motion_threshold:
            motion_detected = True
            break

    if motion_detected:
        # Motion detected, perform face recognition
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        face_detected = False
        for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                # Face recognized
                name = classNames[matchIndex].upper()

                # Send "1" when face is recognized via HTTP POST
                response = requests.post(esp32_ip, data="1")
                print(f"Match found: {name}, sending '1'")

                face_detected = True

            else:
                # Face not recognized
                name = 'Unknown'
                response = requests.post(esp32_ip, data="0")  # Send "0" for unknown face
                print("No match found, sending '0'")

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            if name == 'Unknown':
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    else:
        # No motion detected, send "2" for intruder detected
        response = requests.post(esp32_ip, data="2")
        print("Intruder detected, sending '2'")

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

    previous_frame = gray

cap.release()
cv2.destroyAllWindows()
         