import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = "/home/arihant/Downloads/camera-calibration-master"
fnames = os.listdir(path)

for fname in fnames:
    if fname[-5:] == ".jpeg":
        img = cv2.imread(path + fname)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,threshold = cv2.threshold(grayscale,15,255,cv2.THRESH_BINARY)
        cv2.imshow('threshold',threshold)
        image,contours,heic = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # finding the contours which contain maximum area
        cnts = sorted(contours, key=lambda x:cv2.contourArea(x), reverse = True)[:1]

        for cnt in cnts:
            # applying the contour approximation to get perfect rectangle 
            epsilon = 0.01*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            print(len(approx))
            img2 = cv2.drawContours(img,approx,-1,(0,255,0),3)
            cv2.imshow('img',img)

            cv2.waitKey()
            cv2.destroyAllWindows()
