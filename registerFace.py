import cv2
import os
import numpy as np
import time

text = [""]
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data():
    faces_data = []
    label = []
    while True:
        if not video_capture.isOpened():
            print("Unable to load Camera")
            pass

        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30
                     )
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.namedWindow('Capture')
        cv2.imshow('Video', frame)

        if len(faces) != 0:
            faces_data.append(gray[y:y + w, x:x + h])
            label.append(1)

        if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= 50:
            break

    return faces_data,label





data, labels = prepare_training_data()
video_capture.release()
cv2.destroyAllWindows()
print( data )
print('total face data', len(data))
print("data:=>", data)
print('total labels', len(labels))
print("label:=>", labels)


if data is not None:
    text.append(input('Input your name for testing\n'))
else:
    print("No face detected ")

face_recognizer_LBPH = cv2.face_LBPHFaceRecognizer.create()

print("Start training the faces data")
face_recognizer_LBPH.train(data, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)


video_capture = cv2.VideoCapture(0)
while True:
    if not video_capture.isOpened():
        print('Unable to load Camera')
        pass

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30,30)
    )


    for (x,y,w,h) in faces:
        label = face_recognizer_LBPH.predict(gray[y:y+w, x:x+h])
        label_text = text[label[0]]

        draw_rectangle(frame , faces[0])
        draw_text(frame, label_text, faces[0][0], faces[0][1] -5 )


    cv2.waitKey(5)
    cv2.imshow('Video', frame)

