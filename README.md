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
