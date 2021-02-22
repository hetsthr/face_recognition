from keras.models import load_model

classifier = load_model('MobileNet/mobilenet_25_40_none/Facial_recogNet.h5')

import os
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
import time

facial_recog_dict = {"[0]" : "Het",
        "[1]" : "Prashant"}

facial_recog_dict_n = {"n0" : "Het",
        "n1" : "Prashant"}

def getRandomImage(path):
    """function loads a random image from a random folder in our test path"""
    img = cv2.imread(path)
    return img
correct = 0
for i in range(0,399):
    path = 'face_data/test/'+str(i)+'.jpeg'
    input_im = getRandomImage(path)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,96,96,3)

    start = time.time()
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis = 1)
    stop = time.time()

    if ( i <= 199 ):
        if (facial_recog_dict[str(res)] == "Het"):
            correct = correct + 1
    else:
        if (facial_recog_dict[str(res)] == "Prashant"):
            correct = correct + 1

    #print("Prediction:",facial_recog_dict[str(res)])
    print("Latency:",stop-start)

print("Accuracy: ",correct/4)
