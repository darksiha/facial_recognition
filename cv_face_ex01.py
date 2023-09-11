import cv2
import sys

cascade_file = "/home/blud/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

image_file = "imgs/frame_img/image2.jpg"
image = cv2.imread(image_file)
image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=1, minSize=(150,150))

if len(face_list) > 0:
    print(face_list)
    color = (0, 0, 255)
    for face in face_list:
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=0)
    cv2.imwrite("imgs/Detect_test02.jpg", image)
else:
    print("no face")