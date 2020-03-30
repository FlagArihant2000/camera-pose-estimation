import cv2
import numpy as np

img = cv2.imread('test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

size = img.shape
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

# 2D image points
image_points = np.array([(283,145),(333,145),(284,194),(333,194)], dtype = "float32")

# 3D model points
model_points = np.array([(0.0,0.0,0.0),(60.0,0.0,0.0),(0.0,60.0,0.0),(60.0,60.0,0.0)])
#model_points = np.zeros((10*8,1,3),np.float32)
#model_points[:,:,:2] = np.mgrid[0:10, 0:8].T.reshape(-1,1,2)
# Initialization of camera intrinsic parameters, obtained from the camera calibration github repository. 

camera_matrix = np.array([[1.43131504e+03,0.00000000e+00,6.42552941e+02],[0.00000000e+00,1.42801915e+03,3.49581293e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]])

dist_coeff = np.array([[-4.31340433e-02,-1.61386185e-01,-1.07692904e-03,-2.66596162e-03,4.16181240e+00]])

ret, corners = cv2.findChessboardCorners(gray, (10,8), None)
corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
img = cv2.drawChessboardCorners(img, (10,8), corners2, ret)
ret, rvecs, tvecs = cv2.solvePnP(model_points,image_points,camera_matrix,dist_coeff, flags = cv2.SOLVEPNP_ITERATIVE)
#rvecs, tvecs, inliers = cv2.solvePnPRansac(model_points,image_points,camera_matrix,dist_coeff, flags = cv2.SOLVEPNP_ITERATIVE)
 


print(rvecs)
print(tvecs)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


