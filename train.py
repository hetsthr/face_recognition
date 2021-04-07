import tensorflow as tf
import numpy as np
import train_model
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

DATA_DIR = './face_data'
NUM_CLASSES = 2
BATCH_SIZE = 10 
IMG_SIZE = 32
CHANNELS = 1
MODEL_NAME = 'mobilenet'
EPOCHS = 50
MODEL_DIR = MODEL_NAME+'_'+str(IMG_SIZE)+'/model_tf/'
TRAIN_SAMPLES = 600
VALIDATION_SAMPLES = 0.2*TRAIN_SAMPLES
if(CHANNELS == 1):
    COLOR_MODE = 'grayscale'
    MODEL_DIR = MODEL_NAME+'_gray'+str(IMG_SIZE)+'/model_tf/'
else:
    COLOR_MODE = 'rgb'

def plot_hist(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title(MODEL_NAME)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(["train", "validation"])
    plt.show()


data_dir = DATA_DIR

train_data_dir = data_dir + '/train/'
validation_data_dir = data_dir + '/val/'
num_classes = NUM_CLASSES

train_datagen = ImageDataGenerator(
    rotation_range = 45,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    horizontal_flip = True,
    fill_mode = 'nearest')

validation_datagen = ImageDataGenerator()

batch_size = BATCH_SIZE

img_rows, img_cols = IMG_SIZE,IMG_SIZE 

model = train_model.get_model(MODEL_NAME, IMG_SIZE, CHANNELS, NUM_CLASSES)

model.summary()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_rows, img_cols),
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = COLOR_MODE)

validation_generator = validation_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_rows, img_cols),
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = COLOR_MODE)

checkpoint = ModelCheckpoint(
    'kernel_vs_latency/'+MODEL_NAME+'_gray_'+str(IMG_SIZE)+'.h5',
    monitor = 'val_loss',
    mode = 'min',
    save_best_only = True,
    verbose = 1)

earlystop = EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0,
    patience = 30,
    verbose = 1,
    restore_best_weights = True)

callbacks = [checkpoint]

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = RMSprop(lr = 0.0001),
    metrics = ['accuracy'])

nb_train_samples = TRAIN_SAMPLES
nb_validation_samples = VALIDATION_SAMPLES

epochs = EPOCHS

history = model.fit_generator(
    train_generator,
    steps_per_epoch = np.ceil((nb_train_samples/batch_size)-1),
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = np.ceil((nb_validation_samples/batch_size)-1))

model.save(MODEL_DIR)
plot_hist(history)


