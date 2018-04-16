from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalMaxPooling2D
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras.callbacks import ModelCheckpoint
#from tensorflow.python.ops import nn
from .train import train_model, plot_train_results, predict_image

import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import random


class SegNetShallow:
	def __init__(self, input_shape, num_classes):
		self.model = self.__create_segnet(input_shape, num_classes)
		self.train_history = []
		self.max_pool_indices = []

	#def __MaxPooling2DWithIndices(self, )
	
	def __create_encoder(self, model, input_shape=None):
		self.__conv_and_bn(model, 64, input_shape=input_shape)
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))
		self.__conv_and_bn(model, 128)
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))
		self.__conv_and_bn(model, 256)
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))
		model.add(Dropout(0.25))
		self.__conv_and_bn(model, 512)
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))
		model.add(Dropout(0.25))

		return model

	def __conv_and_bn(self, model, conv_size, k_size=3, input_shape=None):
		if input_shape is not None:
			model.add(Conv2D(conv_size, (k_size, k_size), padding='same', input_shape=input_shape))
		else:
			model.add(Conv2D(conv_size, (k_size, k_size), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

	def __create_segnet(self, input_shape, num_classes):
		model = Sequential()
		
		# encoder
		self.__create_encoder(model, input_shape)
		
		# decoder
		model.add(UpSampling2D((2, 2)))
		self.__conv_and_bn(model, 512)
		model.add(Dropout(0.25))

		model.add(UpSampling2D((2, 2)))
		self.__conv_and_bn(model, 256)
		model.add(Dropout(0.25))

		model.add(UpSampling2D((2, 2)))
		self.__conv_and_bn(model, 128)

		model.add(UpSampling2D((2, 2)))
		self.__conv_and_bn(model, 64)

		model.add(Conv2D(num_classes, (1, 1), padding='valid'))

		model.add(Reshape((input_shape[0]*input_shape[1], num_classes),
						input_shape = (input_shape[0], input_shape[1], num_classes)))

		model.add(Activation('softmax'))
		
		model.summary()

		return model

	def train(self, train_gen, num_train_samples, test_gen, num_test_samples, weights_dir=osp.join(os.getcwd(),'weights'),
				weights_file=None, batch_size=1, epochs=1, optimizer=RMSprop, lr=0.001):
		
		self.train_history = train_model(self.model, train_gen, num_train_samples, test_gen, 
									num_test_samples, weights_dir, weights_file, batch_size, 
									epochs, optimizer, lr)


	def plot_results(self):
		plot_train_results(self.train_history)

	def predict(self, processed_img, weights=None):
		predict_image(self.model, processed_img, weights)