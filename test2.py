import time
import picamera

with picamera.PiCamera() as camera:
    camera.resolution = (1024, 768)
    camera.framerate = 30
    camera.start_preview()
    time.sleep(2)
    camera.capture_sequence(['imgs/frame_img/image1.jpg',
                             'imgs/frame_img/image2.jpg',
                             'imgs/frame_img/image3.jpg',
                             'imgs/frame_img/image4.jpg',
                             'imgs/frame_img/image5.jpg'],
                            use_video_port=True)