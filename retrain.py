from keras.applications import MobileNet
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'face_data'
num_classes = 2


img_rows, img_cols = 96, 96

model = MobileNet(weights = 'imagenet',
        include_top = False,
        input_shape = (img_rows, img_cols, 3))

'''
for (i, layer) in enumerate(model.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)
'''

for layer in model.layers:
    layer.trainable = False

for (i, layer) in enumerate(model.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)

def lw(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

FC_Head = lw(model, num_classes)
model = Model(inputs = model.input, outputs=FC_Head)
print(model.summary())

