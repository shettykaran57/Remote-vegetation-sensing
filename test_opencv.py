# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image
imagePath = r"/home/karan/Remote-vegetation-sensing/project/sundarbans_data/sample.tiff"
img = cv2.imread(imagePath)

hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	
cv2.imshow('Original image',img)
cv2.imshow('HSV image', hsvImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
