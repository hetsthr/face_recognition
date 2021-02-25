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
#cap = cv2.VideoCapture(0)
num = 200
while (num < 400): 
    file_r = './face_data/test/test'+str(num)+'.jpeg'
    img = cv2.imread(file_r)
    cv2.imwrite('./face_data/test/'+str(num)+'.jpeg', img)
    num = num + 1
    cv2.waitKey(10)
    '''
    ret, img = cap.read()
    img = cv2.resize(img, (96,96))
    
    gray = img
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 2)
    for (x,y,w,h) in faces:
#        print (x,y,w,h)
        file = './face_data/test/'+str(num)+'.jpeg'
        aa = gray[y:y+30, x:x+25]
        aa = cv2.resize(aa, (32,32))
        cv2.imshow('image', aa)
        cv2.imwrite(file, aa)
        cv2.waitKey(10)
        num = num + 1
#        print (x, " ", y, " ", w, " ", h)
#        cv2.waitKey(10)
    '''
    '''
    file = './face_data/test/'+str(num)+'.jpeg'
    cv2.imwrite(file, img)
    cv2.imshow('image', img)
    num = num + 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''
