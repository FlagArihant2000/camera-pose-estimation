import cv2
import  numpy as np
import math


def gray_code(n):
                  
        # creating a list1 which will initialize by 0 ,1
        list1 = ['0', '1']
        # list2 is  reverse of list1 
        list2 = list1.copy()
        list2.reverse()
        k =0
        # lengthx is length of every unique gray code  
        lengthx = len(list1[0]) - 1
        

        if n > 1:
            while lengthx < n:

                for j in range(len(list1)):
                # adding 0 to every element of list1 and 1 in every element of list2
                    list1[j] = '0' + list1[j]
                    list2[j] = '1' + list2[j]
                #  list1 is replaced concatenate of  both the list 
                list1 = list1 + list2
                # similary list2 is reverse of list1
                list2 = list1.copy()
                list2.reverse()
                lengthx = len(list1[0])
        return list1

cv2.namedWindow("images", cv2.WINDOW_NORMAL)
# taking input fromm user that heigth and width
h,w = input('enter the dimension').split()
h = int(h)
w = int(w)
k=0
# n is number frame 
n =math.log(w,2)
n = math.ceil(n)        
# generating the gray code for horizontal 
gray_h = gray_code(n)
# making images of generated gray code
for i in range(n):
    image = np.zeros([h,w],np.uint8)

    for j in range(h):
        if gray_h[j][i] == '1':
            image[j,:] = np.array(255,np.uint8)
    cv2.imshow('images',image)
    cv2.imwrite(str(k)+'.png',image) 
    k= k +1
          
    cv2.imshow('images',image)
    cv2.waitKey(0)
    
# similary for   vertical strips
# n is number frame 
n =math.log(h,2)
n = math.ceil(n)        
gray_v = gray_code(n)   
for i in range(n):
    image = np.zeros([h,w],np.uint8)

    for j in range(w):
        if gray_v[j][i] == '1':
            image[:,j] = np.array(255,np.uint8)
    cv2.imshow('images',image)
    cv2.imwrite(str(k)+'.png',image) 
    k= k +1
    cv2.waitKey(0)

cv2.destroyAllWindows()
