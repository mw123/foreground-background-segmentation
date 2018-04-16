# Foreground-Background-Segmentation

In this project, I had set up to implement different semantic segmentation architectures from scratch and train on the SUNRGBD dataset. 

# Motivation

The motivation is two-folds. First motivation is to learn how to build deep networks. Second reason has to do with my own research in urban search and rescue (USAR) robots. One big problem in USAR research is lack of image data. To mitigate this problem, I try to use semantic segmentation to segment out objects of interest and transform the background in order to generate new image data.

# Implementation

In this repository I did my implementation of SegNet. 

[Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." IEEE transactions on pattern analysis and machine intelligence 39.12 (2017): 2481-2495.]

Although I have used other repositories as reference, such as https://github.com/divamgupta/image-segmentation-keras, but I did not reuse other code.

Design decisions:
	- data preprocessing: 
		- subtract mean per channel 
		- huge memory saving by storing image paths not the images
	- variable image input size support
	- depth shift and scale rather than raw scaling

#Requirements: 
	- python 3
	- SUNRGBD data: 
		- This repo has made the data easier to extract :https://github.com/ankurhanda/sunrgbd-meta-data
		- extract the 13 classes labels


TODO:
	- better metrics
	- true max pool indices