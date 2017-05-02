import unittest
import cv2
import matplotlib.pyplot as plt
from img_transformer.filters import Filters
from img_transformer.image_operation import ImageOperator

IO = ImageOperator()
filters = Filters()
IO.image('amelie.jpg')
inImage = IO.im

# show
plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(inImage, cv2.COLOR_BGR2RGB))
plt.show()

outImage = filters.nashville(inImage)

# show
plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(outImage, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imwrite('amelie_nashville.jpg',outImage)
