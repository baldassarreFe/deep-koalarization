import unittest
import cv2
import os
import matplotlib.pyplot as plt
from img_transformer.filters import Filters
from img_transformer.image_operation import ImageOperator

class ApplyFilts(Filters):

	def filter_names(self):
		names = ('nashville','lomo','gotham')

	def apply_filts(self, filename, names = ('nashville','lomo','gotham')):
		
		out = []		
		IO = ImageOperator()
		filters = Filters()
		IO.image(filename)
		inImage = IO.im

		out.append( filters.nashville(inImage))
		out.append( filters.lomo(inImage))
		out.append( filters.gotham(inImage))

		for i in range(len(names)):
			cv2.imwrite(os.path.join('Transformed', names[i] + '_' + filename ),out[i])

