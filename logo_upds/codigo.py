import numpy as np
import cv2

myhaar=cv2.CascadeClassifier('myhaar.xml')

cam =cv2.VideoCapture('video1.mp4')
while True:
    ret,img=cam.read()
    
    bottles = myhaar.detectMultiScale(img,scaleFactor=1.5,minNeighbors=2,minSize=(24,24),maxSize=(255,255))
    for (x,y,w,h) in bottles:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Deteccion logo UPDS',img)
    k=cv2.waitKey(30) & 0xff
    if k==27:
         break
cap.release()
cv2.destroyAllWindows()