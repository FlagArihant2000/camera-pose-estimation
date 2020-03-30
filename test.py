import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = glob.glob('*.jpg')
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)*56.33
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
for fname in images:
    img = cv2.imread(fname)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grayscale, 15, 255, cv2.THRESH_BINARY)
    _, contours, heic = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:1]
    approx = None
    for cnt in cnts:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
      #############################################################################
        number = np.array([a[0] for a in approx])
        x_cen = 0
        y_cen = 0
        for [x,y] in number:
                    x_cen = x_cen +x
                    y_cen = y_cen +y
        x_cen = x_cen//4
        y_cen = y_cen//4
        #img = cv2.circle(img,(x_cen,y_cen),2 , (0,0,255), -1)
        sorted_x = np.array([[[0,0]],[[0,0]],[[0,0]],[[0,0]]])
        for [x,y] in number:
                    if x <= x_cen and y <= y_cen:
                            sorted_x[0][0] = np.array([x,y])
                    if x <= x_cen and y >= y_cen:
                            sorted_x[1][0] = np.array([x,y])
                    if x >= x_cen and y >= y_cen:
                            sorted_x[2][0] = np.array([x,y])
                    if x >= x_cen and y <= y_cen:
                            sorted_x[3][0] = np.array([x,y])

        approx = np.array(sorted_x)
        print(approx)                               

    #########################################################################
  
    world=np.array([[[0.,0.]], [[0.,600.]],[[800.,600.]], [[800.,0.]]])
    H, _ = cv2.findHomography(approx, world, cv2.RANSAC, 3.)
    dst = cv2.warpPerspective(img, H, (800,600))
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,7),None)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #print(approx)
    _ = cv2.drawChessboardCorners(dst, (9,7), corners, ret)
    cv2.imshow('img', dst)
    #cv2.imshow('org', img)
    cv2.waitKey(0)
    objpoints.append(objp)
    imgpoints.append(corners)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (800,600),None,None)
print(mtx, dist)
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints)
    tot_error += error
print("total error: ", tot_error / len(objpoints))
"""
cam=np.array([[[500,132]],[[117,171]],[[92,369]],[[525,357]]])

world=np.array([[[800.,0.]],[[0.,0.]],[[0.,600.]],[[800.,600.]]])
#mtx =np.array([[854.5479,0,320.9312],[0,844.7475,257.66],[0,0,1]])
#dist =np.array([[9.65074162e-01, -1.12439712e+02,  2.33101392e-02  ,5.38326031e-03 ,3.44503370e+03]])
flag, rvec, tvec = cv2.solvePnP(world, cam, mtx, dist)
print(rvec)
print(tvec)
R, _ = cv2.Rodrigues(rvec)
H = np.eye(3)
H[:,:2] = R[:,:2]
H[:,2] = tvec.transpose()[0]
H = np.matmul(mtx, H)
H = np.linalg.inv(H)
u_, v_, w_ = H@np.array([117.,171.])
print(u_/w_, v_/w_)
# print(dist)
# mean_error = 0"""
