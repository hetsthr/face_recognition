import tensorflow as tf
import time
from os import listdir
from os.path import isfile, join
import numpy as np

img_dir = 'face_data/test/'
MODEL_DIR = './kernel_vs_latency/'
CHANNELS = 1
IMG_SIZE = 32
MODELS = ['cnn', 'dnn', 'ds_cnn', 'mobilenet']

labels_dict = {"[0]" : "het", "[1]" : "prashant"}

def evaluate_model(interpreter):
    correct = 0
    avg = 0
    for i in range(0, 200):
        image = next(iter(dataset_list))
        label = str(image).split('/')[2].split('.')[1]
        image = tf.io.read_file(image)
        image = tf.io.decode_png(image, channels=CHANNELS)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        input_im = tf.expand_dims(image, 0)

        input_data = np.array(input_im, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        start = time.time()
        interpreter.invoke()
        stop = time.time()
        
        avg = avg + (stop-start)

        output_data = interpreter.get_tensor(output_details[0]['index'])
        res = np.argmax(output_data, axis = 1)
        #print("Label= ", label, end=' ')
        #print("Prediction = ", res)
        if(label == labels_dict[str(res)]):
            correct = correct + 1
    latency = avg/10
    acc = correct
    return acc, latency


for MODEL_NAME in MODELS:
    if CHANNELS == 1:
        MODEL_NAME = MODEL_NAME + '_gray_32_k4_float.tflite'
    else:
        MODEL_FOLDER = MODEL_NAME + '_rgb'

    MODEL_PATH = MODEL_DIR+'/'+MODEL_NAME
    interpreter = tf.lite.Interpreter(model_path = MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #input_shape = input_details[0]['shape']
    #print(input_shape)

    dataset_list = tf.data.Dataset.list_files(img_dir + '*')
    acc = 0
    latency = 0
    for i in range(0,5):
        a, l = evaluate_model(interpreter)
        acc = acc + a
        latency = latency + l
    print(MODEL_NAME,'\t', CHANNELS, '\t', IMG_SIZE, '\t', acc, '\t', latency/5) 
