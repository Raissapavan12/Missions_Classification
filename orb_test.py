import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('/activity/5e29e0a388c52b472dfdd976.jpg',0)          
img2 = cv.imread('/activity/5e29e05d061f535d6fbab6rf.jpg',0)

cv.imshow('image1', img1)
cv.imshow('image2', img2)
cv.waitKey(0) & 0xFF
cv.destroyAllWindows()

# Initiate SIFT detector
orb = cv.ORB()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.knnMatch(des1,des2, k=2)

distance