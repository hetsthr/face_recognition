import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('models/MODEL_TF')
model_no_quant_tflite = converter.convert()

open('models/MODEL_NO_QUANT.tflite','wb').write(model_no_quant_tflite)
