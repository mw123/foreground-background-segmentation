import os.path as osp
import os
import matplotlib.pyplot as plt
import numpy as np
import random

from tensorflow.python.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint

def train_model(model, train_gen, num_train_samples, test_gen, num_test_samples, weights_dir=osp.join(os.getcwd(),'weights'),
			weights_file=None, batch_size=1, epochs=1, optimizer=RMSprop, lr=0.001):		

	model.compile(loss="categorical_crossentropy", optimizer=optimizer(lr=lr), metrics=['accuracy'])		

	if weights_file is not None:
		model.load_weights(weights_file)

	checkpoint = ModelCheckpoint(osp.join(weights_dir,'SegNet.{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.h5'), 
									monitor='val_acc', verbose=1, save_best_only=True)

	train_history = model.fit_generator(
		train_gen, 
		steps_per_epoch = num_train_samples//batch_size,
		epochs= epochs,
		validation_data = test_gen,
		validation_steps = num_test_samples//batch_size,
		callbacks = [checkpoint])
		#self.model.save_weights(osp.join(weights_dir,'SegNet_final.h5'))
	return train_history

def plot_train_results(train_history):
	# plot accuracy
	plt.plot(train_history.history['acc'])
	plt.plot(train_history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# plot accuracy
	plt.plot(train_history.history['loss'])
	plt.plot(train_history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

#def visualize_layers(self):
#	layer_dict = dict([(layer.name, layer) for layer in self.model.layers])

def predict_image(model, processed_img, weights=None):
	if weights is not None:
		model.load_weights(weights)
	img_tensor = np.expand_dims(processed_img, axis=0)
	prob = model.predict(img_tensor)[0]
	num_classes = model.output_shape[-1]
	output_dim = np.sqrt(model.output_shape[-2]).astype(int)
	prob = prob.reshape((output_dim, output_dim, num_classes)).argmax(axis=2)
	
	seg_result = np.zeros((output_dim, output_dim, 3)).astype('uint8')
	# randomly assign colors to every class, except '__background__' class
	colors = [(random.randint(0,150),random.randint(50,150),random.randint(150,255)) for _ in range(num_classes)]
	for cls_ in range(1, num_classes):
		seg_result[:,:,0] += (np.where(prob == cls_, 1, 0)*colors[cls_][0]).astype('uint8')
		seg_result[:,:,1] += (np.where(prob == cls_, 1, 0)*colors[cls_][1]).astype('uint8')
		seg_result[:,:,2] += (np.where(prob == cls_, 1, 0)*colors[cls_][2]).astype('uint8')
	plt.imshow(seg_result)
	plt.show()