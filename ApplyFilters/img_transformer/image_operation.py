import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class ImageOperator(object):

	def __init__(self):
		pass

	def image(self, filename):
		self.im = cv2.imread(filename)

	def brightness_contrast(self, img, alpha = 1.0, beta = 0):
		img_contrast = img * (alpha)
		img_bright = img_contrast + (beta)
		# img_bright = img_bright.astype(int)
		img_bright = stats.threshold(img_bright,threshmax=255, newval=255)
		img_bright = stats.threshold(img_bright,threshmin=0, newval=0)
		img_bright = img_bright.astype(np.uint8)
		return img_bright

	def channel_enhance(self, img, channel, level=1):
		if channel == 'B':
			blue_channel = img[:,:,0]
			# blue_channel = (blue_channel - 128) * (level) +128
			blue_channel = blue_channel * level
			blue_channel = stats.threshold(blue_channel,threshmax=255, newval=255)
			img[:,:,0] = blue_channel
		elif channel == 'G':
			green_channel = img[:,:,1]
			# green_channel = (green_channel - 128) * (level) +128
			green_channel = green_channel * level
			green_channel = stats.threshold(green_channel,threshmax=255, newval=255)
			img[:,:,0] = green_channel
		elif channel == 'R':
			red_channel = img[:,:,2]
			# red_channel = (red_channel - 128) * (level) +128
			red_channel = red_channel * level
			red_channel = stats.threshold(red_channel,threshmax=255, newval=255)
			img[:,:,0] = red_channel
		img = img.astype(np.uint8)
		return img

	def hue_saturation(self, img_rgb, alpha = 1, beta = 1):
		img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
		hue = img_hsv[:,:,0]
		saturation = img_hsv[:,:,1]
		hue = stats.threshold(hue * alpha ,threshmax=179, newval=179)
		saturation = stats.threshold(saturation * beta,threshmax=255, newval=255)
		img_hsv[:,:,0] = hue
		img_hsv[:,:,1] = saturation
		img_transformed = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
		return img_transformed

	#Use interpolate instead of look uptable which was so 10 years ago
	def histogram_matching(self, img, matching_img, number_of_bins = 255):
		img_res = img.copy()
		for d in range(img.shape[2]):
			img_hist, bins = np.histogram(img[:,:,d].flatten(), number_of_bins, normed=True)
			matching_img_hist, bins = np.histogram(matching_img[:,:,d].flatten(), number_of_bins, normed=True)
			#print bins[:-1]

			cdf_img = img_hist.cumsum()
			cdf_img = (255 * cdf_img / cdf_img[-1]).astype(np.uint8) #normalize

			cdf_match = matching_img_hist.cumsum()
			cdf_match = (255 * cdf_match / cdf_match[-1]).astype(np.uint8) #normalize

			im2 = np.interp(img[:,:,d].flatten(), bins[:-1], cdf_img)
			im3 = np.interp(im2, cdf_match, bins[:-1])

			img_res[:,:,d] = im3.reshape((img.shape[0]),img.shape[1])

		return img_res




if __name__ == '__main__':
	IO = ImageOperator()
	IO.image('horror.jpg')
	source_img = IO.im
	IO.image('amelie.jpg')
	matching_img = IO.im
	img = IO.histogram_matching(source_img, matching_img)
	cv2.imshow('image', img)
	cv2.waitKey(0)


