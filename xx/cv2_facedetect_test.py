import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
def facedetect():
    
    face_cascade = cv2.CascadeClassifier("/home/blud/haarcascade_frontalface_default.xml")
    
    try:
        cap = cv2.VideoCapture(0)
    except:
        print("Video Error")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray           = 분석될 이미지 또는 영상 프레임
        # scaleFactor    = 이미지에서 얼굴 크기가 서로 다른 것을 보상해주는 값
        # minNeighbors   = 얼굴 사이의 최소 간격( 픽셀 )
        # minSize        = 얼굴의 최소 크기
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Faces', (x-5, y-5), font, 0.5, (255,0,0),2)
            
        cv2.imshow('frame', frame)
        # cv2.imwrite("imgs/testframe.png", frame)
    facedetect()