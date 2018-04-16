from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalMaxPooling2D
from tensorflow.python.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras.callbacks import ModelCheckpoint
#from tensorflow.python.ops import nn
from .train import train_model, plot_train_results, predict_image


import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import random


class SegNet:
	def __init__(self, input_shape, num_classes):
		self.model = self.__create_segnet(input_shape, num_classes)
		self.train_history = []
		self.max_pool_indices = []

	#def __MaxPooling2DWithIndices(self, )
		
	def __create_encoder(self, input_shape=None):
		model = Sequential()

		model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(128, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Conv2D(256, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(256, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(256, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(512, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(512, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(512, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(512, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))
		#model.summary()

		# encoder is vgg16 architecture with pre-trained Imagenet weights
		WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
		weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
          						WEIGHTS_PATH_NO_TOP,
          						cache_subdir='models',
          						file_hash='6d6bbae143d832006294945121d1f1fc')

		model.load_weights(weights_path)

		for layer in model.layers[:5]:
			layer.trainable = False

		return model

	def __conv_and_bn(self, model, conv_size, k_size=3):
		model.add(Conv2D(conv_size, (k_size, k_size), padding='same'))
		model.add(BatchNormalization())
		#model.add(Activation('relu'))
		return model

	def __create_segnet(self, input_shape, num_classes):
		model = Sequential()
		
		# encoder
		encoder = self.__create_encoder(input_shape)
		# starting from pre-trained vgg16 model, we will interleave 
		# with BatchNorm layers where appropriate
		for i, layer in enumerate(encoder.layers):
			model.add(layer)

			if layer.__class__.__name__ == 'Conv2D':
				model.add(BatchNormalization())

		model.add(Dropout(0.5))
		
		# decoder
		model.add(UpSampling2D((2, 2)))
		model = self.__conv_and_bn(model, 512)
		model = self.__conv_and_bn(model, 512)
		model = self.__conv_and_bn(model, 512)
		model.add(Dropout(0.5))

		model.add(UpSampling2D((2, 2)))
		model = self.__conv_and_bn(model, 512)
		model = self.__conv_and_bn(model, 512)
		model = self.__conv_and_bn(model, 512)
		model.add(Dropout(0.5))

		model.add(UpSampling2D((2, 2)))
		model = self.__conv_and_bn(model, 256)
		model = self.__conv_and_bn(model, 256)
		model = self.__conv_and_bn(model, 256)
		model.add(Dropout(0.5))

		model.add(UpSampling2D((2, 2)))
		model = self.__conv_and_bn(model, 128)
		model = self.__conv_and_bn(model, 128)

		model.add(UpSampling2D((2, 2)))
		model = self.__conv_and_bn(model, 64)
		model = self.__conv_and_bn(model, 64)

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