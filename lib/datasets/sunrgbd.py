import os
import numpy as np
import itertools
import random
from .frame_data import Frame_Data

class SUNRGBD:
	DATA_PATH = 'data/sunrgbd-data/'
	NUM_TRAIN_IMAGES = 5285 #TODO: make this a variable instead of constant
	NUM_TEST_IMAGES = 5050 #TODO: make this a variable instead of constant
	CROP_SIZE = 416

	CLASSES = ('__background__',
				'Bed', 'Books', 'Ceiling', 'Chair', 'Floor', 'Furniture',
				'Objects', 'Picture', 'Sofa', 'Table', 'TV', 'Wall', 'Window')
	NUM_CLASSES = len(CLASSES) 
	
	def __init__(self):
		self.train_data = []
		self.test_data = []
		
		self.__per_channel_mean = []
		self.__per_channel_std = []
		self.__depth_mean = 0
		self.__depth_std = 0
		# first compute mean and std across dataset
		self.__load_data_path()

	def __load_data_path(self):
		data_loc = os.path.join(os.getcwd(), SUNRGBD.DATA_PATH)

		train_rgb_path = os.path.join(data_loc, 'sunrgbd_train_rgb')
		train_depth_path = os.path.join(data_loc, 'sunrgbd_train_depth')
		train_labels_path = os.path.join(data_loc, 'train13labels')

		test_rgb_path = os.path.join(data_loc, 'sunrgbd_test_rgb')
		test_depth_path = os.path.join(data_loc, 'sunrgbd_test_depth')
		test_labels_path = os.path.join(data_loc, 'test13labels')

		for i in range(SUNRGBD.NUM_TRAIN_IMAGES):
			train_rgb_img = os.path.join(train_rgb_path, 'img-{:06d}.jpg'.format(i+1)) 
			train_depth_img = os.path.join(train_depth_path, '{}.png'.format(i+1))
			train_gnd_truth = os.path.join(train_labels_path, 'img13labels-{:06d}.png'.format(i+1))

			img = Frame_Data(train_rgb_img, train_depth_img, train_gnd_truth)
			self.train_data.append(img)

		for i in range(SUNRGBD.NUM_TEST_IMAGES):
			test_rgb_img = os.path.join(test_rgb_path, 'img-{:06d}.jpg'.format(i+1)) 
			test_depth_img = os.path.join(test_depth_path, '{}.png'.format(i+1))
			test_gnd_truth = os.path.join(test_labels_path, 'img13labels-{:06d}.png'.format(i+1))

			img = Frame_Data(test_rgb_img, test_depth_img, test_gnd_truth)
			self.test_data.append(img)

		self.__compute_mean_std()

	def display_image(self, ith_img, img_set):
		if (ith_img == 0 or (img_set=='train' and ith_img > SUNRGBD.NUM_TRAIN_IMAGES) 
				or (img_set=='test' and ith_img > SUNRGBD.NUM_TEST_IMAGES)):
			print('Please select from images 1-(NUM_TRAIN(TEST)_IMAGES)')
			return None
		return self.train_data[ith_img-1].display_image_label()

	def __compute_mean_std(self):
		rgb_channel_num = 3
		rgb_pixel_num = 0 # total # of pixels in dataset
		sum_per_channel = np.zeros(rgb_channel_num)
		square_per_channel = np.zeros(rgb_channel_num)
		
		depth_pixel_num = 0
		depth_sum = 0
		depth_squared = 0

		print('Processing dataset...\n')

		# first compute per-channel mean
		for img in self.train_data:
			rgb, depth, _ = img.read_frame_data()
			
			rgb_pixel_num += (rgb.size//rgb_channel_num)
			sum_per_channel += np.sum(rgb.astype(np.uint32), axis=(0,1)) #bgr 
			square_per_channel += np.sum(np.square(rgb.astype(np.uint32)), axis=(0,1))

			depth_pixel_num += depth.size
			depth_sum += np.sum(depth.astype(np.uint32))
			depth_squared += np.sum(np.square(depth.astype(np.uint32)))

		self.__per_channel_mean = sum_per_channel/rgb_pixel_num
		self.__per_channel_std = np.sqrt((square_per_channel/rgb_pixel_num) 
											- np.square(self.__per_channel_mean))
		
		self.__depth_mean = depth_sum/depth_pixel_num
		self.__depth_std = np.sqrt((depth_squared/depth_pixel_num) - np.square(self.__depth_mean))
		print('Done.\n')

	def __crop_image(self, img, gnd=None):
		# In order for the network to produce an output that has the same dimensions as the input 
		# image, we need input of size mxm, where m is a multiple of 32. I chose the largest feasible
		# dimensions from the image dataset. With the cropping size 416x416, it is possible to produce five 
		# smaller cuts by cropping each image either from one of the corners or the center. We randomly 
		# choose one among the five images.
		crop_size = SUNRGBD.CROP_SIZE

		img_cuts = []
		# top-left corner
		img_cuts.append(img[0:crop_size,0:crop_size])
		# bottom-left corner
		img_cuts.append(img[img.shape[0]-crop_size:img.shape[0], 0:crop_size])
		# top-right corner
		img_cuts.append(img[0:crop_size, img.shape[1]-crop_size:img.shape[1]])
		# bottom-right corner
		img_cuts.append(img[img.shape[0]-crop_size:img.shape[0], img.shape[1]-crop_size:img.shape[1]])
		# center cut
		img_cuts.append(img[img.shape[0]//2-crop_size//2:img.shape[0]//2+crop_size//2, 
								img.shape[1]//2-crop_size//2:img.shape[1]//2+crop_size//2])						

		rand_idx = np.random.randint(0,len(img_cuts))
		
		if gnd is not None:
			gnd_cuts = []
			gnd_cuts.append(gnd[0:crop_size,0:crop_size])
			gnd_cuts.append(gnd[img.shape[0]-crop_size:img.shape[0], 0:crop_size])
			gnd_cuts.append(gnd[0:crop_size, img.shape[1]-crop_size:img.shape[1]])
			gnd_cuts.append(gnd[img.shape[0]-crop_size:img.shape[0], img.shape[1]-crop_size:img.shape[1]])
			gnd_cuts.append(gnd[img.shape[0]//2-crop_size//2:img.shape[0]//2+crop_size//2, 
									img.shape[1]//2-crop_size//2:img.shape[1]//2+crop_size//2])

			return [img_cuts[rand_idx], gnd_cuts[rand_idx]]
		
		return img_cuts[rand_idx]

	def __format_gnd(self, gnd):
		num_classes = SUNRGBD.NUM_CLASSES
		seg_labels = np.zeros((gnd.shape[0], gnd.shape[1], num_classes))
		
		for cls_ in range(num_classes):
			seg_labels[:, :, cls_] = np.where(gnd == cls_, 1, 0)
		
		seg_labels = np.reshape(seg_labels, (gnd.shape[0]*gnd.shape[1], num_classes))#.astype(int)
		
		return seg_labels

	def preprocess_image(self, img, gnd=None):
		if len(img.shape) == 3: #rgb
			rgb = img.astype(np.float64)
			rgb[:,:,0] -= self.__per_channel_mean[0]
			rgb[:,:,1] -= self.__per_channel_mean[1]
			rgb[:,:,2] -= self.__per_channel_mean[2]
			rgb = rgb[:,:,::-1]
			rgb = rgb.astype(np.float32)

			crop = self.__crop_image(rgb, gnd)

		elif len(img.shape) == 2: # depth
			# first normalize depth image by dividing std, then scale and shift to have 
			# the same mean and std as the rgb dataset
			depth_norm = (img.astype(np.float64) - self.__depth_mean)/self.__depth_std
			depth_scaled = depth_norm * np.mean(self.__per_channel_std) #+ np.mean(self.__per_channel_mean)
			depth_clipped = np.clip(depth_scaled, (0-np.mean(self.__per_channel_mean)), 
											(255-np.mean(self.__per_channel_mean)))
			depth_clipped = depth_clipped.astype(np.float32)
							
			crop = self.__crop_image(depth_clipped, gnd)
		return crop

	def data_generator(self, img_set, img_type, batch_size=1, shuffle=True):
		if img_set == 'train':
			data = self.train_data
		elif img_set == 'test':
			data = self.test_data
		
		if shuffle:
			data = random.sample(data, len(data))

		data_iter = itertools.cycle(data)
		while True:
			X_batch = []
			Y_batch = []
			
			for _ in range(batch_size):
				rgb, depth, gnd = next(data_iter).read_frame_data()

				if img_type == 'rgb':
					crop = self.preprocess_image(rgb, gnd)	
				elif img_type == 'depth':
					crop = self.preprocess_image(rgb, gnd)
					
				X_batch.append(crop[0])
				Y_batch.append(self.__format_gnd(crop[1]))
			
			yield np.array(X_batch), np.array(Y_batch)
