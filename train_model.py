import tensorflow as tf

from keras.layers import *
from keras.models import Sequential, Model

def get_model(model_name, img_size, channels, num_classes=1, shallow=True, alpha=0.25):
    model = Sequential()
    if(channels == 1):
        model.add(Input((img_size, img_size, 1)))
    else:
        model.add(Input((img_size, img_size, 3)))

    if(model_name == 'mobilenet'):
        model.add(Convolution2D(int(32 * alpha), (4,4), strides=(2,2), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(DepthwiseConv2D(3, (1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(64 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(DepthwiseConv2D(3, (2,2), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(128 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(DepthwiseConv2D(3, (1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(128 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(DepthwiseConv2D(3, (2,2), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(256 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(DepthwiseConv2D(3, (1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(256 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(DepthwiseConv2D(3, (2,2), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(512 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        if not shallow:
              for _ in range(5):
                      model.add(DepthwiseConv2D(3, (1,1), padding="same", use_bias=False))
                      model.add(BatchNormalization())
                      model.add(Activation('relu'))
                      model.add(Convolution2D(int(512 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
                      model.add(BatchNormalization())
                      model.add(Activation('relu'))

        model.add(DepthwiseConv2D(3, (2,2), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(1024 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(DepthwiseConv2D(3, (1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(1024 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

    if(model_name == 'cnn'):
        model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
        model.add(MaxPool2D(2,2))
        model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
        model.add(MaxPool2D(3,3))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
    
    if(model_name == 'ds_cnn'):
        model.add(Convolution2D(32, (4,4), strides=(2,2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(DepthwiseConv2D(3, (2,2), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(1024 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(2,2))
        model.add(DepthwiseConv2D(3, (1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(int(1024 * alpha), (1,1), strides=(1,1), padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(3,3))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

    if(model_name == 'dnn'):
        model.add(Conv2D(32, (4,4), padding='same', activation='relu'))
        model.add(MaxPool2D(2,2))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(2,2))
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(2,2))
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(2,2))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

         
    return model
