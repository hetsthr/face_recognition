# Face Recognition
The goal is to re-train MobileNet for facial recognition, convert the trained model for microcontrollers and test the accuracy / latency / power parameters.

## Testing the model before quantization:
Val\_Accuracy = 100%

Test\_Accuracy = 86%

Avg. Latency = 200ms

## Dataset:
Image Size: 96x96x3

Contains two categories: Het / Prashant

Train Images: 242

Validation Images : 158

Test Images : 200

## Current Issues:
* Segmentation Fault when trying to use tf.lite.TFLiteConverter.from\_saved\_model() on RPi

## Results for MobileNet:
### Training for different Alpha (depth multiplier)
1. Alpha = 0.5 (Default Weights = 'imagenet', 10 epochs)

* .h5 Model = 7 MB
* tflite\_model = 3.2 MB
* tflite\_quant = model\_cc = 1 MB
* val\_acc = 100%

2. Alpha = 0.25 (Default Weights = 'imagenet', 10 epochs)

* .h5 Model = 2.2 MB
* tflite\_model = 902 KB
* tflite\_quant = model\_cc = 335.2 KB
* val\_acc = 100%

3. Alpha = 0.125 (Default Weights = None, 10 epochs)

* .h5 Model = 948 KB
* tflite\_model = 268 KB
* tflite\_quant = model\_cc = 130 KB
* val\_acc = 51%

4. Alpha = 0.125 (Default Weights = None, 40 epochs)

* .h5 Model = 948 MB
* tflite\_model = 268 KB
* tflite\_quant = model\_cc = 130 KB
* val\_acc = 98.5%

