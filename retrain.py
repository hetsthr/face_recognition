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

batch_size = 5

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

model = MobileNet(weights = 'imagenet',
        include_top = False,
        input_shape = (img_rows, img_cols, 3))

'''
for (i, layer) in enumerate(model.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)
'''

for layer in model.layers:
    layer.trainable = False

'''
for (i, layer) in enumerate(model.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)
'''

def lw(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
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
        patience = 3,
        verbose = 1,
        restore_best_weights = True)

callbacks = [earlystop, checkpoint]

model.compile(loss = 'categorical_crossentropy',
        optimizer = RMSprop(lr = 0.001),
        metrics = ['accuracy'])

nb_train_samples = 232
nb_validation_samples = 128

epochs = 5
batch_size = 5

history = model.fit_generator(
        train_generator,
        steps_per_epoch = np.ceil((nb_train_samples*0.8/batch_size)-1),
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = np.ceil((nb_validation_samples*0.8/batch_size)-1))

# Converting the Model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
