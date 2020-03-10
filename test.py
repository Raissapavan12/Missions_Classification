#https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
#https://github.com/llvll/imgcluster/blob/master/imgcluster.py

from skimage import metrics
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#reading image
img1 = cv.imread('/activity/5e29e0a388c52b472dfdd976.jpg', -1)
img2 = cv.imread('/activity/5e29e0c471679a0cfb7333a9.jpg', -1)

cv.imshow('image1', img1)
cv.imshow('image2', img2)
cv.waitKey(0) & 0xFF
cv.destroyAllWindows()

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	return err

#até aqui temos um número 0/

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = metrics.structural_similarity(imageA, imageB, multichannel=True)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()