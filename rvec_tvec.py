import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('1279.jpg')
plt.imshow(img)
plt.show()
cam=np.array([[[236.54318,168.40318]],[[253.52243,166.82744]],[[357.573,265.10233]],[[375.78625,264.51984]]])

world=np.array([[[56.33,56.33,0.]],[[112.66,56.33,0.]],[[450.64,394.31,0.]],[[506.97,394.31,0.]]])
mtx =np.array([[825.76,0,310.95],[0,824.58,259.77],[0,0,1]])
dist =np.array([[0.03246184, -0.05476432, 0.00295698, -0.00672755, 0.63743764]])
flag, rvec, tvec = cv2.solvePnP(world, cam, mtx, dist)
R, _ = cv2.Rodrigues(rvec)
H = np.eye(3)
H[:,:2] = R[:,:2]
H[:,2] = tvec.transpose()[0]
H = np.matmul(mtx, H)
H = np.linalg.inv(H)
u_, v_, w_ = H@np.array([236.54318,168.40318, 1.])
print(u_/w_, v_/w_)
