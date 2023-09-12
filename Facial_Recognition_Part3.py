import cv2
import numpy as np
import time
from os import listdir
from os.path import isfile, join


##### 여기서부터는 Part2.py와 동일 
data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data, Labels = [], []
for i, files in enumerate(onlyfiles):
    #if not '.jpg' in files: # 확장자가 jpg가 아닌 경우 무시
    #    continue
    if images is None:  # 이미지 파일로 불러오지 못 했다면 무시
        continue
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is None:
        continue    
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
if len(Labels) == 0:
    print("There is no data to train.")
    exit()
Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")
#### 여기까지 Part2.py와 동일 

#### 여긴 Part1.py와 거의 동일 
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달
#### 여기까지 Part1.py와 거의 동일 
#카메라 열기 
cap = cv2.VideoCapture(0)

while True:
    #카메라로 부터 사진 한장 읽기 
    ret, frame = cap.read()
    # 얼굴 검출 시도 
    image, face = face_detector(frame)
    try:       
        #검출된 사진을 흑백으로 변환 
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #위에서 학습한 모델로 예측시도
        result = model.predict(face)
        #result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다. 
        if result[1] < 500:
            #????? 어쨋든 0~100표시하려고 한듯 
            confidence = int(100*(1-(result[1])/300))
            # 유사도 화면에 표시 
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        #75 보다 크면 동일 인물로 간주해 UnLocked! 
        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            #cv2.imshow('Face Cropper', image)
            #본인 확인 완료 및 유사도 출력
            print (confidence, "% Confidence it is user. 본인이 맞습니다.")
            
            break
        else:
           #75 이하면 타인.. Locked!!! 
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            #cv2.imshow('Face Cropper', image)
            #본인 외 다른 사람 사진 혹은 얼굴일 때, 혹은 유사도가 75 이하인 경우
            print ("본인 아님")
    # 이미지를 찾을 수 없을 경우 예외 발생
    except:
        #얼굴 검출 안됨 
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        #cv2.imshow('Face Cropper', image)
        print ("얼굴 검출 실패")
        pass
    # Enter 키를 누르면 루프가 중단됩니다. 13 is the Enter Key
    if cv2.waitKey(1)==13:
        break

#릴리스 및 destroyAllWindows 
cap.release()
cv2.destroyAllWindows()

