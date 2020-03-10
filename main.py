import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
from scipy import stats
import time
from pathlib import Path
import os

rootPath = str(Path('../../').resolve())
imgPath = rootPath + '/activity'

orb = cv2.ORB_create() #creating ORB object

def listImages():
    files = os.listdir(imgPath)
    return files

def calcTrimMeanDistance(im1Name, im2Name):
    im1 = cv2.imread(imgPath + '/' + im1Name)
    im2 = cv2.imread(imgPath + '/' + im2Name)
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_bf = bf.match(des1, des2)

    matches = sorted(matches_bf, key=lambda x: x.distance)

    distances = np.zeros(len(matches))
    for i in range(len(matches)):
        distances[i] = matches[i].distance

    return stats.trim_mean(distances, 0.25)
    # return totalDistance

def findBestMatch(images, index):
    sourceImg = images[index]
    distances = np.full(len(images), 10000)
    matchedImage = ''
    minDistance = 10000
    for i in range(len(images)):
        if(i != index):
            dist = calcTrimMeanDistance(sourceImg, images[i])
            if(dist < minDistance):
                minDistance = dist
                matchedImage = images[i]

    # distances.sort()
    im1 = cv2.imread(imgPath + '/' + sourceImg)
    im2 = cv2.imread(imgPath + '/' + matchedImage)
    # cv2.imshow('source', im1)
    # cv2.imshow('matched', im2)
    print(sourceImg)
    print(matchedImage)
    sideBysSideImg = np.concatenate((im1, im2), axis=1)
     cv2.imshow('image', sideBysSideImg)
     cv2.waitKey(0)



files = listImages()
for i in range(len(files)):
    findBestMatch(files, i)

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html