import cv2

video = cv2.VideoCapture(0)

frame_count = 30

for i in range(frame_count):
    ret, frame = video.read()
    if ret:
        image_path = f"faces/image_{i}.jpg"
        cv2.imwrite(image_path, frame)

video.release()
