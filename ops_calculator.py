from keras.models import load_model
import sys
MODEL_PATH = str(sys.argv[1])
print(MODEL_PATH)
model = load_model(MODEL_PATH)
ops = 0
print("-----------------------")
for layer in model.layers:
	if hasattr(layer, 'kernel_size'):
		#print(layer.input_shape)
		a1 = layer.output_shape[1]*layer.output_shape[2]*layer.output_shape[3]
		a2 = layer.kernel_size[0]*layer.kernel_size[1]*layer.input_shape[3]
		ops += a1 * a2
print(ops/1000000)
print("----------------------")
