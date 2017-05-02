from image_operation import ImageOperator
import filter
import border
import vignette

from filter import *
from border import *
from vignette import *

class Filters(ImageOperator, Border, Vignette):

	def nashville(self, img, hue = 1, saturation = 1.5, contrast = 1.5, brightness = -30):
		img = self.hue_saturation(img, hue, saturation)
		img = self.brightness_contrast(img, contrast, brightness)
		return img


	def lomo(self, filename):
		self.execute("convert {filename} -channel R -level 33% -channel G -level 33% {filename}", filename)
		self.vignette(filename)

	def gotham(self, img, hue = 1, saturation = 0.1, contrast = 1.3, brightness = -20):
		img = self.hue_saturation(img, hue, saturation)
		img = self.brightness_contrast(img, contrast, brightness)
		return img

	def claredon(self, img, hue = 1.2, saturation = 1.4, contrast = 1.4, brightness = - 20):
		img = self.hue_saturation(img, hue, saturation)
		img = self.brightness_contrast(img, contrast, brightness)
		img = self.channel_enhance(img, "B", 1.3)
		return img
	
	def kelvin(self, filename):
		self.execute("convert \( {filename} -auto-gamma -modulate 120,50,100 \) \( -size {width}x{height} -fill 'rgba(255,153,0,0.5)' -draw 'rectangle 0,0 {width},{height}' \) -compose multiply {filename}", filename);

	def nash2(self, filename):
		self.colortone(filename,'#222b6d', 50, 0);
		self.colortone(filename,'#f7daae', 120, 1);
		self.execute("convert {filename} -contrast -modulate 100,150,100 -auto-gamma {filename}", filename);
	
	def toaster(self, filename):
		self.colortone(filename,'#330000', 50, 0)
		self.execute("convert {filename} -modulate 150,80,100 -gamma 1.2 -contrast -contrast {filename}", filename);
		self.vignette(filename,'none', 'LavenderBlush3');
		self.vignette(filename,'#ff9966', 'none');
		self.border(filename, 'white')

