import tensorflow as tf
from keras.applications import MobileNet
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

train_data_dir = 'face_data/train/'
validation_data_dir = 'face_data/validation/'

num_classes = 2

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 4

img_rows, img_cols = 96, 96

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = 'categorical')

model = MobileNet(#weights=None  #uncomment when alpha=0.125
        include_top = False,
        input_shape = (img_rows, img_cols, 3),
        alpha=0.25)

'''
for (i, layer) in enumerate(model.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)
'''

for layer in model.layers:
    layer.trainable = True

'''
for (i, layer) in enumerate(model.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)
'''

def lw(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    #top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(64, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

FC_Head = lw(model, num_classes)
model = Model(inputs = model.input, outputs=FC_Head)
#print(model.summary())

checkpoint = ModelCheckpoint("Facial_recogNet.h5",
        monitor = "val_loss",
        mode = "min",
        save_best_only = True,
        verbose = 1)

earlystop = EarlyStopping(monitor = "val_loss",
        min_delta = 0,
        patience = 7,
        verbose = 1,
        restore_best_weights = True)

callbacks = [earlystop, checkpoint]

model.compile(loss = 'categorical_crossentropy',
        optimizer = RMSprop(lr = 0.001),
        metrics = ['accuracy'])

nb_train_samples = 242
nb_validation_samples = 138

epochs = 10
batch_size = 4

history = model.fit_generator(
        train_generator,
        steps_per_epoch = np.ceil((nb_train_samples*0.8/batch_size)-1),
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = np.ceil((nb_validation_samples*0.8/batch_size)-1))

model.save("models/MODEL_TF")

# Converting the Model to tflite
# Did not work on RPi when tried for the last time.

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model_no_quant = converter.convert()

with open('model_no_quant.tflite', 'wb') as f:
    f.write(tflite_model_no_quant)

# Converting the model to tflite with quantization
img_dir = 'face_data/test'
def rep_data_gen():
    dataset_list = tf.data.Dataset.list_files(img_dir + '*')
    for i in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [96, 96])
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = rep_data_gen

converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

model_quant_tflite = converter.convert()

open('models/model_quant.tflite','wb').write(model_quant_tflite)

#!apt-get update && apt-get -qq install xxd
!xxd -i models/model_quant.tflite > models/model_data.cc

REPLACE_TEXT = 'models/model_quant.tflite'.replace('/','_').replace('.','_')
!sed -i 's/'{REPLACE_TEXT}'/g_model/g' 'models/model_data.cc'
!tail models/model_data.cc
