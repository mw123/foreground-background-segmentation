import os
import os.path as osp
import setup
from models import SegNetShallow
from datasets import SUNRGBD

import matplotlib.pyplot as plt
import cv2
import random
import numpy as np

def main():
	data_path = 'data/sunrgbd-data/sunrgbd_test_rgb'
	weights_path = 'weights/SegNet.10-1.62-0.465.h5'
	out_path = 'out'

	
	src_img = input("Welcome! Let's do some image background transformation!\n\n" +
			"Okay! First please first provide me an image you would like\n" +
    		" to change. Either provide me the full path to the image, or\n" +
			" select a repository image ID from 1-5050.\n")

	dataset = SUNRGBD()
	try: # see if user input an image ID
		img_id = int(src_img)
		src_img_path = osp.join(os.getcwd(), data_path, 'img-{:06d}.jpg'.format(img_id))
		
		orig = cv2.imread(src_img_path)
		orig = dataset.preprocess_image(orig)

	except ValueError as e: # expecting full path to image specified
		src_img_path = src_img
		orig = cv2.imread(src_img_path)
		orig = orig[:,:,::-1]

	print("Displaying your image...")
	plt.imshow(orig)
	plt.show()

	input_shape = (dataset.CROP_SIZE, dataset.CROP_SIZE, 3)
	model = SegNetShallow(input_shape, dataset.NUM_CLASSES)

	probs = model.predict(orig, weights_path)

	user_satified = False
	# until user satisfied with result, keep randomly select an image
	# from database and fill pixels in the original image it is predicted
	# to be background 
	while not user_satified:
		print('Trasforming your image...')
		
		rand_id = random.randint(1,dataset.NUM_TEST_IMAGES)
		backgr = osp.join(os.getcwd(), data_path, 'img-{:06d}.jpg'.format(rand_id))
		backgr = cv2.imread(backgr)
		mask = np.where(probs == 0)
		
		for ix in range(len(mask[0])):
			orig[mask[0][ix],mask[1][ix]] = backgr[mask[0][ix],mask[1][ix]]
		plt.imshow(orig)
		plt.show()
		user_input = input("Are you satisfied with the result?[Y/n]")
		if user_input == 'y' or user_input == 'Y':
			user_satified = True
			cv2.imwrite(osp.join(out_path, 'transformation.png'), orig)

	
if __name__ == '__main__':
    main()