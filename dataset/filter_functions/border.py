from filter import Filter

class Border(Filter):
	
	def border(self,filename,  color = 'black', width = 20):
		self.execute("convert {filename} -bordercolor {color} -border {bwidth}x{bwidth} {filename}", filename, 
			color = color,
			bwidth = width
		)
