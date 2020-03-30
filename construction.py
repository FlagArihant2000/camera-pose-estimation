import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d
path = "/home/arihant/Downloads/camera-calibration-master"
path = os.chdir(path)
def order_points(points):
                len_ = len(points)
                a = points[0][0]
                b = points[len_-1][0]
                if a[0]>b[0] and a[1]>b[1]:
                       points =  points[::-1]                
                return points
def arrange_points(points):
    number = np.array([a[0] for a in points])
    x_cen = 0
    y_cen = 0
    for [x, y] in number:
        x_cen = x_cen + x
        y_cen = y_cen + y
    x_cen = x_cen // 4
    y_cen = y_cen // 4
    # img = cv2.circle(img,(x_cen,y_cen),2 , (0,0,255), -1)
    sorted_x = np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]])
    for [x, y] in number:
        if x <= x_cen and y <= y_cen:
            sorted_x[0][0] = np.array([x, y])
        if x <= x_cen and y >= y_cen:
            sorted_x[1][0] = np.array([x, y])
        if x >= x_cen and y >= y_cen:
            sorted_x[2][0] = np.array([x, y])
        if x >= x_cen and y <= y_cen:
            sorted_x[3][0] = np.array([x, y])
    points = np.array(sorted_x)
    return  points
def decode_graycode():
        def inversegrayCode(n): 
            inv = 0
            while(n): 
                inv = inv ^ n
                n = n >> 1
            return inv
        print(os.getcwd())
        path = "/home/arihant/Downloads/camera-calibration-master/graycode.py"
        path = os.chdir(path)
        fname = os.listdir(path)
        img1= cv2.imread('40.png',0)
        img2 = cv2.imread('41.png',0)
        mask = np.where( img2-img1 <50 ,np.uint8(0),np.uint8(255))
        image,contour,hiearchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contour,key=lambda x:cv2.contourArea(x) , reverse = True)[:1]
        approx = None
        for cnt in cnts:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx  = arrange_points(approx)   
        #print(approx) 
        mask = np.zeros(img2.shape,np.uint8)
        mask[approx[0][0][1]:approx[2][0][1],approx[0][0][0]:approx[2][0][0]] = np.uint8(255)        
        ansx = np.zeros(mask.shape,np.uint16)
        ansy = np.zeros(mask.shape,np.uint16) 
        for i in range(0,10):
                 name1= str(i)+'.png'
                 rough1 = np.zeros(mask.shape,np.uint8)  
                 name2 = str(i+20)+'.png'         
                 img1= cv2.imread(name1,0)
                 img2 =cv2.imread(name2,0)
                 a = mask & img1
                 b = mask & img2
                 rough1 = np.where( a>b ,1,0)  
                 ansx = ansx + (rough1)*(2**(9-i))
                 rough2 = np.zeros(mask.shape,np.uint8)
                 name1= str(i+10)+ '.png'
                 name2 = str(i+30)+'.png'         
                 img1= cv2.imread(name1,0)
                 img2 =cv2.imread(name2,0)
                 a  =mask & img1
                 b = mask & img2
                 rough2 = np.where( a>b,1,0)                 
                 ansy =ansy + (rough2)*(2**(19-i))
                 rough1= np.where(a>b,np.uint8(255),np.uint8(0))
                 rough2= np.where(a>b,np.uint8(255),np.uint8(0))
                 cv2.imshow('img1',rough1.astype(np.uint8))
                 cv2.imshow('img2',rough2.astype(np.uint8))
                 cv2.waitKey(1)   
        for row in range(720):
               for col in range(1280):
                        ansy[row][col] = inversegrayCode(ansy[row][col])
                        ansx[row][col] = inversegrayCode(ansx[row][col])
        cv2.destroyAllWindows()
        return ansx,ansy, mask
###################################################################################################################
path = '/home/arihant/Downloads/camera-calibration-master'
path = os.chdir(path)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
fname = os.listdir(path)
objp = np.zeros((9 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)*50.33

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

i = 0
for name in fname:   
    if name[-4:] == '.png':
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
        if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                corners2 = order_points(corners2)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, (9, 7), corners2, ret)
                cv2.imshow('camera', img)
                cv2.waitKey(1)
cv2.destroyAllWindows()  
ret,mtx_c,dist_c,rvec_c,tvec_c = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
print('camera matrix')
print(mtx_c, ret)
path ='/home/arihant/Downloads/camera-calibration-master'
os.chdir(path)
img = cv2.imread('camera_pose.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,corners = cv2.findChessboardCorners(gray ,(9,7),None)
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
imgpoints  = np.array(corners,np.float32)
flag,rvec_c,tvec_c = cv2.solvePnP(objp,imgpoints,mtx_c,dist_c)

H_c = np.zeros((3,3))
world_points = []
R,_ = cv2.Rodrigues(rvec_c)
H_c[:,:2] = R[:,:2]
H_c[:,2] = tvec_c.transpose()[0]
H_c = mtx_c @ H_c
H_c = np.linalg.inv(H_c)
print(H_c)
pose = cv2.imread('/home/saurabh/camera_calibration/24-9-2019/chess.png')
gray = cv2.cvtColor(pose,cv2.COLOR_BGR2GRAY)
ret,corners = cv2.findChessboardCorners(gray,(15,11),None)
print(ret)
if ret:
        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners = order_points(corners)    
        _ = cv2.drawChessboardCorners(pose,(15,11),corners,ret)
        img_points= np.array(corners,np.float32)    
cv2.imshow('img',pose) 
cv2.waitKey(1)
cv2.destroyAllWindows()
path = '/home/saurabh/camera_calibration/24-9-2019/projector_calibration'
path = os.chdir(path)
fname = os.listdir()
projector_points = []
world_points = []
for name in fname: 
#     print(name)
     point = []
     if name[-4:] == '.png':
        img1 = cv2.imread(name)
        img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img = ~img
        blur = cv2.GaussianBlur(img,(5,5),-1)
        img = cv2.addWeighted(img,1.5,blur,-0.5,0,img)
        ret,corners2 = cv2.findChessboardCorners(img,(15,11),None)
        #print(ret)
        if ret :
            corners2 = cv2.cornerSubPix(img,corners2,(11,11),(-1,-1),criteria)
            corners2 = order_points(corners2)
#print(corners2)
            _ = cv2.drawChessboardCorners(img1,(15,11),corners2,ret)
            for corner in corners2:
                
                b = cv2.convertPointsToHomogeneous(corner)
                b= b.reshape(3,1)
                a = H_c @ b
                point.append(a) 
            points = cv2.convertPointsFromHomogeneous(np.array(point))
            world_points.append(points)
            projector_points.append(img_points)
            #print(world_points)
            cv2.imshow('img',img1)
            cv2.waitKey(1)
world = []
for p in world_points:
    #print(p)
    w = np.array([np.array([i[0][0],i[0][1],0],np.float32) for i in p],np.float32)
    world.append(w)
cv2.destroyAllWindows()
ret,mtx_p,dist_p,rvec_p,tvec_p = cv2.calibrateCamera(world,projector_points,(800,600),None,None)
print(mtx_p, ret)
path ='/home/saurabh/camera_calibration/26-9-2019'
os.chdir(path)
img = cv2.imread('projector_pose.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = ~gray
blur = cv2.GaussianBlur(gray,(5,5),-1)
gray = cv2.addWeighted(gray,1.5,blur,-0.5,0,gray)        
ret,corners = cv2.findChessboardCorners(gray ,(15,11),None)
print(ret)
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
corners = order_points(corners)
imgpoints  = np.array(corners,np.float32)
point= []
for corner in corners:
                
                b = cv2.convertPointsToHomogeneous(corner)
                b= b.reshape(3,1)
                a = H_c @ b
                point.append(a) 
                
points = cv2.convertPointsFromHomogeneous(np.array(point)) 
w = np.array([np.array([np.array([i[0][0],i[0][1],0],np.float32) for i in points],np.float32)])
print(w.shape)
flag,rvec_p,tvec_p = cv2.solvePnP(w,imgpoints,mtx_p,dist_p)
H_p = np.zeros((3,4))
world_points = []
R,_ = cv2.Rodrigues(rvec_p)
H_p[:,:3] = R[:,:3]

H_p[:,3] = tvec_p.transpose()[0]
print(H_p)
H_p = mtx_p @ H_p
H_c = np.zeros((3,4))
R,_ = cv2.Rodrigues(rvec_c)
#print(R)
H_c[:,:3] = R[:,:3]
H_c[:,3] = tvec_c.transpose()[0]
print(H_c)
H_c = mtx_c @ H_c
ansx,ansy,mask = decode_graycode()
print(ansx.shape[:2])
plt.imshow(ansx)
plt.show()
plt.imshow(ansy)
plt.show()
img = cv2.imread('41.png')
cv2.imshow('img',img)
cy, cx = np.where(mask == 255)
#print(cy,cx)
values = img[cy,cx]
#print(img[300,300])
print(values)
b,g,r = values.reshape(-1,3).T
py, px = ansy[cy, cx], ansx[cy, cx]
cv2.waitKey(0)
cv2.destroyAllWindows()
camera = np.ones((2,cx.shape[0]))
projector = np.ones((2,cx.shape[0]))

camera[0,:] = cx
camera[1,:] = cy
projector[0,:] = px
projector[1,:] = py

print(H_p.shape, H_c.shape)
points = cv2.triangulatePoints(H_c,H_p,camera,projector)
# points = points.reshape
# print(corners.shape, points.shape)
# print(points.T.reshape(-1,1,4), points.T)
world_p = cv2.convertPointsFromHomogeneous(points.T.reshape(-1,1,4))        
x,y,z =  world_p.reshape(-1,3).T
print(x,y,z)

plt.scatter(x,y)
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection="3d")
ax.scatter3D(x, y, z, c=z, cmap='hsv');
plt.show()
import open3d as o3d
xyzrbg = np.zeros((np.size(x),6))
pcd = o3d.geometry.PointCloud()
xyzrbg[:,0] = np.array(x)
xyzrbg[:,1] = np.array(y)
xyzrbg[:,2] = np.array(z)
xyzrbg[:,3] = np.array(r)/255
xyzrbg[:,4] = np.array(g)/255
xyzrbg[:,5] = np.array(b)/255
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyzrbg[:,:3])
o3d.io.write_point_cloud('/home/saurabh/camera_calibration/26-9-2019/data.ply',pcd)
pcd_load = o3d.io.read_point_cloud('/home/saurabh/camera_calibration/26-9-2019/data.ply')
img = o3d.io.read_image("41.png")
print(img)
o3d.visualization.draw_geometries([pcd_load])
