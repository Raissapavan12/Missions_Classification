"""
    a) Loads Image
    b) Detect Keypoints & Descriptor
    c) Matches
"""

import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
import time
from pathlib import Path

#
# Load Images

rootPath = str(Path('../../').resolve())

im1 = cv2.imread(rootPath + '/images/5b0ea48b621ceba8a4c9f9b6_before.jpeg')
im2 = cv2.imread(rootPath + '/images/5b0ea48b621ceba8a4c9f9b6_after.jpeg')
startTime = time.time()
#
# ORB Feature Detector
#orb =  cv2.DescriptorExtractor_create("BRIEF") #cv2.AKAZE_create() #Need opencv3. some bug with python binding of orb detector.
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(im1, None)
kp2, des2 = orb.detectAndCompute(im2, None)
print('len(kp1) : ', len(kp1), '    des1.shape : ', des1.shape)
print('len(kp2) : ', len(kp2), '    des2.shape : ', des2.shape)
print('Time (ms) : ', (time.time() - startTime)*1000.)

#
# Draw Keypoints
# im1_keypts = cv2.drawKeypoints(im1, kp1, None )
# im2_keypts = cv2.drawKeypoints(im2, kp2, None )
# cv2.imshow('keypts1', im1_keypts)
# cv2.imshow('keypts2', im2_keypts)


#
# Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_bf = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches_bf, key = lambda x:x.distance)

#
# FLANN Matcher
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1.astype('float32'),des2.astype('float32'),k=2)
# matches = sum( matches, [] )
aboveThreshold = []
for m in matches:
    if m.distance < 40:
        aboveThreshold.append(m)

print(aboveThreshold)

# Draw first 10 matches.
img3 = cv2.drawMatches(im1, kp1, im2, kp2, aboveThreshold, None)

cv2.imshow('matches', img3)
cv2.waitKey(0)

# print('Time (ms) : ', (time.time() - startTime)*1000.)
#
# Draw Matches
# im_matches = cv2.drawMatches(im1,kp1,  im2,kp2,  matches, None)
# cv2.imshow( 'matches', im_matches)
# cv2.waitKey(0)
