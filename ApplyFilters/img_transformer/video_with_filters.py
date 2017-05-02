import cv2
from image_operation import ImageOperator
from filters import Filters


F = Filters()
video_path = "/home/vionlabs/Documents/weilun_thesis/die_another_day/big_bang_1.mp4"
frame_index = 3000
length = 3000
cap = cv2.VideoCapture(video_path)
mask_image = cv2.imread('horror.jpg')

while(cap.isOpened() and frame_index < length+frame_index):
	cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_index)
	ret, frame = cap.read()
	frame_index += 1
	if frame.any():
		# img = F.nashville(frame)
		img = F.hist_match(frame, mask_image)
		cv2.imshow("frame", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break