import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
%matplotlib inline

img1 = cv2.imread('..\\Camerman_G_0.05.jpg',0)
img2 = cv2.imread('..\\Cameraman_SandP_0.08.jpg',0)

def medianFilter(img,size):
    m,n=img.shape
    
    # Traverse the image. For every 3X3 or 5x5 area,find the median of the pixels and replace the ceter pixel by the median 
    img_new = np.zeros([m, n]) 
    len=int((size-1)/2)
    median=int((size*size)/2)
    temp=list()
    for i in range(len, m-len): 
        for j in range(len, n-len): 
            for k in range (i-len,i+len+1):
                for w in range (j-len,j+len+1):
                    temp.append(img[k,w])
            
            temp = sorted(temp) 
            img_new[i, j]= temp[median] 
            temp.clear()

    img_new = img_new.astype(np.uint8) 
    return img_new

def meanFilter(img,size):
    m,n=img.shape
    
    # Traverse the image. For every 3X3 or 5x5 area,find the mean of the pixels and replace the ceter pixel by the mean 
    img_new = np.zeros([m, n]) 
    len=int((size-1)/2)
    median=int((size*size)/2)
    temp=list()
    for i in range(len, m-len): 
        for j in range(len, n-len): 
            for k in range (i-len,i+len+1):
                for w in range (j-len,j+len+1):
                    temp.append(img[k,w])
            
            value = sum(temp) 
            img_new[i, j]= value/size
            temp.clear()

    img_new = img_new.astype(np.uint8) 
    return img_new

def Min_MaxFilter(img,size):
    m,n=img.shape
    
    # Traverse the image. For every 3X3 or 5x5 area,find the minimum of the pixels and replace the ceter pixel by the min value 
    img_new = np.zeros([m, n]) 
    len=int((size-1)/2)
    median=int((size*size)/2)
    temp=list()
    for i in range(len, m-len): 
        for j in range(len, n-len): 
            for k in range (i-len,i+len+1):
                for w in range (j-len,j+len+1):
                    temp.append(img[k,w])
            
            value = min(temp) 
            img_new[i, j]= value
            temp.clear()
    
    # Traverse the image. For every 3X3 or 5x5 area,find the maximum of the pixels and replace the ceter pixel by the max value
    img_new1 = np.zeros([m, n])
    for i in range(len, m-len): 
        for j in range(len, n-len): 
            for k in range (i-len,i+len+1):
                for w in range (j-len,j+len+1):
                    temp.append(img_new[k,w])
            
            value = max(temp) 
            img_new1[i, j]= value
            temp.clear()

    img_new1 = img_new1.astype(np.uint8) 
    
    return img_new1

plt.figure(figsize=(15,15))

img1_meanfilter3 = meanFilter(img1,3)
img1_meanfilter3=cv2.cvtColor(img1_meanfilter3.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,1)
plt.title("3*3 mean filter on image 1")
plt.imshow(img1_meanfilter3)

img1_meanfilter5 = meanFilter(img1,5)
img1_meanfilter5=cv2.cvtColor(img1_meanfilter5.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,2)
plt.title("5*5 mean filter on image 1")
plt.imshow(img1_meanfilter5)

img1_filter3 = medianFilter(img1,3)
img1_filter3=cv2.cvtColor(img1_filter3.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,3)
plt.title("3*3 median filter on image 1")
plt.imshow(img1_filter3)

img1_filter5 = medianFilter(img1,5)
img1_filter5=cv2.cvtColor(img1_filter5.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,4)
plt.title("5*5 median filter on image 1")
plt.imshow(img1_filter5)

img1_minmaxfilter3 = Min_MaxFilter(img1,3)
img1_minmaxfilter3=cv2.cvtColor(img1_minmaxfilter3.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,5)
plt.title("3*3 min-max filter on image 1")
plt.imshow(img1_minmaxfilter3)

img1_minmaxfilter5 = Min_MaxFilter(img1,5)
img1_minmaxfilter5=cv2.cvtColor(img1_minmaxfilter5.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,6)
plt.title("5*5 min-max filter on image 1")
plt.imshow(img1_minmaxfilter5)

plt.figure(figsize=(15,15))

img1_meanfilter3 = meanFilter(img2,3)
img1_meanfilter3=cv2.cvtColor(img1_meanfilter3.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,1)
plt.title("3*3 mean filter on image 2")
plt.imshow(img1_meanfilter3)

img1_meanfilter5 = meanFilter(img2,5)
img1_meanfilter5=cv2.cvtColor(img1_meanfilter5.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,2)
plt.title("5*5 mean filter on image 2")
plt.imshow(img1_meanfilter5)

img1_filter3 = medianFilter(img2,3)
img1_filter3=cv2.cvtColor(img1_filter3.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,3)
plt.title("3*3 median filter on image 2")
plt.imshow(img1_filter3)

img1_filter5 = medianFilter(img2,5)
img1_filter5=cv2.cvtColor(img1_filter5.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,4)
plt.title("5*5 median filter on image 2")
plt.imshow(img1_filter5)

img1_minmaxfilter3 = Min_MaxFilter(img2,3)
img1_minmaxfilter3=cv2.cvtColor(img1_minmaxfilter3.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,5)
plt.title("3*3 min-max filter on image 2")
plt.imshow(img1_minmaxfilter3)

img1_minmaxfilter5 = Min_MaxFilter(img2,5)
img1_minmaxfilter5=cv2.cvtColor(img1_minmaxfilter5.astype("uint8"),cv2.COLOR_BGR2RGB)
plt.subplot(3,2,6)
plt.title("5*5 min-max filter on image 2")
plt.imshow(img1_minmaxfilter5)

