import math
from filter import Filter

class Vignette(Filter):
	def vignette(self,filename, color_1 = 'none', color_2 = 'black', crop_factor = 1.5):
		crop_x = math.floor(self.size(filename)[0] * crop_factor)
		crop_y = math.floor(self.size(filename)[1] * crop_factor)
		 
		self.execute("convert \( {filename} \) \( -size {crop_x}x{crop_y} radial-gradient:{color_1}-{color_2} -gravity center -crop {width}x{height}+0+0 +repage \) -compose multiply -flatten {filename}",filename, 
			crop_x = crop_x,
			crop_y = crop_y,
			color_1 = color_1,
			color_2 = color_2,
		)
