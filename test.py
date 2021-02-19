from keras.models import load_model

classifier = load_model('Facial_recogNet.h5')

import os
import numpy as np
from os import listdir
from os.path import isfile, join

facial_recog_dict = {"[0]" : "Het",
        "[1]" : "Prashant"}

facial_recog_dict_n = {"n0" : "Het",
        "n1" : "Prashant"}

def getRandomImage(path):
    """function loads a random image from a random folder in our test path"""
    
