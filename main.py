import cv2
import threading
import time

cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
video_capture = cv2.VideoCapture(0)

while True:
    if not video_capture.isOpened():
        print('Unable to load camera')
        pass

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("faces->", faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # cv2.namedWindow("Image")
    # cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or time.process_time() >= 10:
        break

    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()
