from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import imutils

faceCascade = cv2.CascadeClassifier('facial_recognition_model.xml')

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

        # show the frame   
        cv2.imshow("face", image)
        rawCapture.truncate(0)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
