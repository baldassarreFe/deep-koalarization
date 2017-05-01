import unittest
import cv2
import os
import matplotlib.pyplot as plt
import shutil 

from filter_functions.filters import Filters
from filter_functions.image_operation import ImageOperator

class ApplyFilts(Filters):

	def apply_filts(self, filename, namesCV = ('nashville', 'gotham', 'claredon'), namesIM = ('lomo', 'kelvin', 'nash2', 'toaster')):
		
		out = []		
		IO = ImageOperator()
		filters = Filters()
		IO.image(filename)
		inImage = IO.im
		
		# OPEN CV FILTERS
		out.append( filters.nashville(inImage))
		IO.image(filename)
		inImage = IO.im
		out.append( filters.gotham(inImage))
		IO.image(filename)
		inImage = IO.im
		out.append( filters.claredon(inImage))
		for i in range(len(namesCV)):
			cv2.imwrite(os.path.join('Transformed', namesCV[i] + '_' + filename ),out[i])

		# IM filters
		filenameDumb = os.path.join('Transformed', namesIM[0] + '_' + filename )
		shutil.copyfile(filename, filenameDumb)
		filters.lomo(filenameDumb)

		filenameDumb = os.path.join('Transformed', namesIM[1] + '_' + filename )
		shutil.copyfile(filename, filenameDumb)
		filters.kelvin(filenameDumb)

		filenameDumb = os.path.join('Transformed', namesIM[2] + '_' + filename )
		shutil.copyfile(filename, filenameDumb)
		filters.nash2(filenameDumb)

		filenameDumb = os.path.join('Transformed', namesIM[3] + '_' + filename )
		shutil.copyfile(filename, filenameDumb)
		filters.toaster(filenameDumb)



