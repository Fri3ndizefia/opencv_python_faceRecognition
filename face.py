import cv2
import os
import numpy as np

subjects = [ "", "Ramiz Raja", "Elvis Presley"]


def detect_face(img):
    # convert the image from color image to gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector
    # Here is using LBP
    # more correct but slower one is Haar
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    face_cascade_Haar = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

    # start detecting
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    print('faces->', faces)
    print('faces len->', len(faces))
    if (len(faces) == 0):
        return None, None

    # If there is only one face
    # get the rectangle area of face
    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):

    # Step-1
    # get the data files directory
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue;

        # Step-2
        # get the label from dir_name
        label = int(dir_name.replace("s", ""))

        # 建立包含当前主题主题图像的目录路径
        subject_dir_path = data_folder_path + "/" + dir_name

        # 获取给定主题目录内的图像名称
        subject_images_names = os.listdir(subject_dir_path)

        # Step-3
        # 浏览每个图片的名称， 阅读图片
        # 检测脸部并将脸部添加到脸部列表
        for image_name in subject_images_names:

            # 忽略掉系统文件
            if image_name.startswith("."):
                continue;

            # 建立图像路径
            image_path = subject_dir_path + "/" + image_name

            #阅读图像
            image = cv2.imread(image_path)

            cv2.imshow("Training on image", image)
            cv2.waitKey(100)

            #侦测脸部
            face, rect = detect_face(image)

            # Step-4
            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces:", len(faces))
print("faces =>", faces)
print("Total labels:", len(labels))
print("labels =>", labels)


# OpenCV 有三个人脸识别器
# EigenFaces 人脸识别器
# FisherFaces 人脸识别器
# LBPH 人脸识别器
face_recognizer_LBPH = cv2.face_LBPHFaceRecognizer.create()
face_recognizer_Eigen = cv2.face_EigenFaceRecognizer.create()
face_recognizer_Fisher = cv2.face_FisherFaceRecognizer.create()

face_recognizer_LBPH.train(faces, np.array(labels))



def draw_rectangle(img, rect):
    (x,y,w,h) = rect
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)



def draw_text(img, text, x, y):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255,0), 2)

def predict(test_img):
    img = test_img.copy()

    face, rect = detect_face(img)

    label = face_recognizer_LBPH.predict(face)
    print('label ->', label)

    label_text = subjects[label[0]]

    draw_rectangle(img, rect)

    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img

print("Predicting images...")
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")

predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

