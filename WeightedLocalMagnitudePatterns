
from skimage import feature
import numpy as np
import cv2

class WeightedLocalMagnitudePatterns:
	def __init__(self, numPoints, radius):

		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		result = np.array((
			[0, 0, 0],
			[0, 0, 0],
			[0, 0, 0]), dtype="int")
		(iH, iW) = image.shape[:2]
		(kH, kW) = result.shape[:2]
		pad = (kW - 1) // 2
		image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
				cv2.BORDER_REPLICATE)
		array = np.array(image)
		image_regeration = np.zeros_like(image)
		print(np.shape(array))

		temp = []
		histogram = []
		#for i in range(0,8):
		#	for j in range(0,8):
		#		print(array[i][j], end = " ")
		#	print()

		#print("___________________________________________________")    

		for y in np.arange(pad, iH + pad):
			for x in np.arange(pad, iW + pad):
				roi = array[y - pad:y + pad + 1, x - pad:x + pad + 1]		
				for i in range(0,3):
					    for j in range(0,3):
					    	if(roi[i][j] > roi[1][1]) :
					    		result[i][j] = abs(roi[i][j]-roi[1][1])
					    	else:
					    		result[i][j] = abs(roi[1][1]-roi[i][j])
					    	temp.append(result[i][j])
					    	#print(result[i][j],end = " ")
					    #print()

				temp.sort(reverse=True)
				temp.pop()
				summation = 0
				for i in range(len(temp)):
					summation += pow(2,i) * temp[i]
				if(summation > 255)	:
					histogram.append(255)
					summation = 255
				else:
					histogram.append(summation)

				roi2 = image_regeration[y - pad:y + pad + 1, x - pad:x + pad + 1]
				roi2[1][1] = summation;
				image_regeration[y - pad:y + pad + 1, x - pad:x + pad + 1] = roi2
				temp.clear()
		lbp = image_regeration;
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))


		hist = hist.astype("float")
		hist /= (hist.sum() + eps)


		return hist
