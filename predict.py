from WeightedLocalMagnitudePatterns.WeightedLocalMagnitudePatterns import WeightedLocalMagnitudePatterns
import numpy as np
import cv2
from imutils import paths
import imutils
import pickle

desc = WeightedLocalMagnitudePatterns(24, 8)


with open("pickle_model.pkl", 'rb') as file:
    pickle_model = pickle.load(file)

for imagePath in paths.list_images("images/testing"):
	
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = pickle_model.predict(hist.reshape(1, -1))

	
	cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

