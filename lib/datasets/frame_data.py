import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Frame_Data:	
	def __init__(self, rgb_path, depth_path, gnd_truth):
		self.rgb_path = rgb_path
		self.depth_path = depth_path
		self.gnd_truth = gnd_truth

	def display_image_label(self):
		rgb = mpimg.imread(self.rgb_path)
		depth = mpimg.imread(self.depth_path)
		gnd_truth = mpimg.imread(self.gnd_truth)

		fig, ax = plt.subplots(1, 3)
		ax[0].set_title('RGB image')
		ax[0].set_axis_off()
		ax[0].imshow(rgb)
		ax[1].set_title('Depth image')
		ax[1].set_axis_off()
		ax[1].imshow(depth, cmap='gray')
		ax[2].set_title('Ground truth')
		ax[2].set_axis_off()
		ax[2].imshow(gnd_truth)  
		plt.show()

	def read_frame_data(self):
		return cv2.imread(self.rgb_path, cv2.IMREAD_COLOR), \
				cv2.imread(self.depth_path, cv2.IMREAD_ANYDEPTH), \
				cv2.imread(self.gnd_truth, cv2.IMREAD_ANYDEPTH)