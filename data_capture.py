# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 23:26:56 2021

@author: Admin
"""

import cv2
import numpy as np

x = 0
y = 0
num = 0
#%%
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while (num < 2000): 
    ret, img = cap.read()
    img = cv2.resize(img, (380,200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 2)
    for (x,y,w,h) in faces:
#        print (x,y,w,h)
        file = './face_data/prashant/'+str(num)+'.png'
        aa = gray[y:y+100, x:x+100]
        cv2.imwrite(file, aa)
        cv2.waitKey(10)
        num = num + 1
#        print (x, " ", y, " ", w, " ", h)
#        cv2.waitKey(10)
    cv2.imshow('image', gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()