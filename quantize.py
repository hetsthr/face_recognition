import os
from keras.models import load_model
import tensorflow as tf
data_dir = './face_data'
img_dir = data_dir + '/test/'
CHANNELS = 1
IMG_SIZE = 32
MODEL_NAME = ['cnn', 'ds_cnn', 'dnn', 'mobilenet']
def rep_data_gen():
    dataset_list = tf.data.Dataset.list_files(img_dir + '*')
    for i in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=CHANNELS)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.expand_dims(image, 0)
        yield [image]

for model in MODEL_NAME:
    if CHANNELS == 1:
        MODEL = model+'_gray_'+str(IMG_SIZE)+'_k4'
    else:
        MODEL = model+'_rgb_'+str(IMG_SIZE)

    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('kernel_vs_latency/'+MODEL+'.h5')

    tflite_float = converter.convert()

    with open(str(MODEL)+'_float.tflite', 'wb') as f:
     f.write(tflite_float)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = rep_data_gen

    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant = converter.convert()

    with open(str(MODEL)+'_quant.tflite', 'wb') as f:
        f.write(tflite_quant)
    
    tflite_model = str(MODEL)+'_quant.tflite'
    cc_model = str(MODEL)+'.cc'
    c1 = 'xxd -i '+tflite_model+' > '+cc_model
    os.system(c1)
    c3 = "tail "+cc_model
    os.system(c3)

