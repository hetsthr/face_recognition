from keras.models import load_model
import tensorflow as tf
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import time

labels_dict = {"[0]" : "het", "[1]" : "prashant"}

img_dir = 'face_data/test/'
dataset_list = tf.data.Dataset.list_files(img_dir + "*")

MODEL_DIR = './kernel_vs_latency'
MODELS = ['cnn','dnn', 'ds_cnn', 'mobilenet']
CHANNELS = 1
IMG_SIZE = 32

def evaluate_model():
	correct = 0
	avg = 0
	for i in range(0,200):
		image = next(iter(dataset_list))
		label = str(image).split('/')[2].split('.')[1]
		image = tf.io.read_file(image)
		image = tf.io.decode_jpeg(image, channels=CHANNELS)
		image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
		image = tf.expand_dims(image, 0)
	
		start = time.time()
		res = np.argmax(classifier.predict(image, 1, verbose = 0), axis = 1)
		stop = time.time()
		avg = avg + (stop-start)
		if(label == labels_dict[str(res)]):
			correct = correct + 1

	latency = avg/10
	acc = correct
	return acc, latency
for MODEL_NAME in MODELS:
	if CHANNELS == 1:
		MODEL_NAME = MODEL_NAME+'_gray_32_k4.h5'
		#MODEL_NAME = MODEL_NAME + "_gray"
	else:
		MODEL_FOLDER = MODEL_NAME + "_rgb"

	MODEL_PATH = MODEL_DIR+"/"+MODEL_NAME

	classifier = load_model(MODEL_PATH)

	acc = 0
	latency = 0 
	for i in range(0,5):
		a, l = evaluate_model()
		acc = acc + a
		latency = latency + l
	print(MODEL_NAME,'\t', CHANNELS, '\t', IMG_SIZE, '\t', acc, '\t', latency/5)

