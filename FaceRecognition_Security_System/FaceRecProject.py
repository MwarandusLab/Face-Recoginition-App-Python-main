import cv2
import numpy as np
import face_recognition
import os
import requests
import threading
import queue
import time  # Import time module for timing control

# Face recognition parameters
path = "ImagesAttendance"
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

classNames.append('Unknown')
print("Loaded Classes:", classNames)


# Encode Faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Only append if encoding is found
            encodeList.append(encode[0])
        else:
            print("Warning: No face found in", img)
    return encodeList


encodeListKnown = findEncodings(images)
print("Encoding Complete")

# ESP32 HTTP Endpoint
esp32_ip = "http://192.168.210.1/status"

# Create a queue for passing frames
frame_queue = queue.Queue()
output_queue = queue.Queue()  # Queue to return processed frames

# Variable to track last request time
last_request_time = 0


def process_faces():
    """Thread for face recognition processing"""
    global last_request_time  # Use global timestamp variable

    while True:
        if not frame_queue.empty():
            img = frame_queue.get()

            # Resize and convert for face recognition
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # Detect faces
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            if not encodesCurFrame:
                print("No face detected in frame")
                output_queue.put(img)  # Put unchanged frame
                continue

            for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        data_to_send = "1"
                    else:
                        name = "Unknown"
                        data_to_send = "2Sms = 0;"

                    # Send request only every 2 seconds
                    current_time = time.time()
                    if current_time - last_request_time >= 2:
                        try:
                            response = requests.post(esp32_ip, data=data_to_send, timeout=2)
                            print(f"Sent '{data_to_send}' to ESP32")
                            last_request_time = current_time  # Update last request time
                        except requests.exceptions.RequestException as e:
                            print(f"ESP32 request failed: {e}")

                # Scale back face location
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Draw rectangle around face
                color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            output_queue.put(img)  # Put processed frame back in queue


# Start the face recognition thread
face_thread = threading.Thread(target=process_faces, daemon=True)
face_thread.start()

# Open Video Capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Send frame to queue for processing
    if frame_queue.qsize() < 2:  # Limit queue size to prevent memory overload
        frame_queue.put(img.copy())

    # Display processed frame if available
    if not output_queue.empty():
        img = output_queue.get()

    cv2.imshow('Webcam', img)  # Display live stream

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
