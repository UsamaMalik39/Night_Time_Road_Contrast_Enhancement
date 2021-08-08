# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:45:28 2020

@author: Malik Usama
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
def calculate_count(count,to_count):
    for i in range(0,256,1):
        count=count+to_count[i]
    return count
    
    
def calculate_cdf(val_array,weight_array,normal_weight):
    count_sum=0
    count_sum=calculate_count(count_sum, weight_array)
    cdf=np.array([0.0]*256)
    
    for i in range(0,256,1):
        normal_weight[i]=weight_array[i] / count_sum
    for i in range(0,256,1):
        cdf[i]=0
        if (i==0):
            cdf[i]=normal_weight[i]
        else:
            for j in range(0,i,1):
                cdf[i]=cdf[i]+normal_weight[j]
    return cdf
def create_plot_arrays(rows,cols,image):
    level_val = np.array([0]*256)
    level_weight=np.array([0]*256)
    for i in range(0,rows,1):
        for j in range(0,cols,1):
            index=image[i][j]
            level_val[index]=index
            level_weight[index]=level_weight[index]+1
    return level_val,level_weight

def showplot(x,y):
    plt.plot(x,y)

normalized_weight_img1=np.array([0.0]*256)
normalized_weight_img2=np.array([0.0]*256)
normalized_weight_img3=np.array([0.0]*256)

path1 ="C:/EME/7th sem/dip/assignment/#1/handout/data/hw1_dark_road_1.jpg"
img1=cv.imread(path1,0)
clahe1 = cv.createCLAHE(clipLimit=4.5, tileGridSize=(10,20))
cl11 = clahe1.apply(img1)
cv.imshow("Clipped Image", cl11)
plt.show()
cv.waitKey(0)


path2 ="C:/EME/7th sem/dip/assignment/#1/handout/data/hw1_dark_road_2.jpg"
img2=cv.imread(path2,0)
clahe2 = cv.createCLAHE(clipLimit=4.5, tileGridSize=(10,20))
cl12 = clahe2.apply(img2)
cv.imshow("Clipped Image", cl12)
plt.show()
cv.waitKey(0)

path3 ="C:/EME/7th sem/dip/assignment/#1/handout/data/hw1_dark_road_3.jpg"
img3=cv.imread(path3,0)
clahe3 = cv.createCLAHE(clipLimit=4.5, tileGridSize=(10,20))
cl13 = clahe3.apply(img3)
cv.imshow("Clipped Image", cl13)
plt.show()
cv.waitKey(0)
 
"size of img_1 is (450,800,3)"
img1_val,img1_weight=create_plot_arrays(450, 800,cl11)



"size of img_2 is (480,640,3)"
img2_val,img2_weight=create_plot_arrays(480, 640,cl12)




"size of img_3 is (450,800,3)"
img3_val,img3_weight=create_plot_arrays(450, 800,cl13)

cdf_img1=calculate_cdf(img1_val, img1_weight,normalized_weight_img1)
cdf_img2=calculate_cdf(img2_val, img2_weight,normalized_weight_img2)
cdf_img3=calculate_cdf(img3_val, img3_weight,normalized_weight_img3)



showplot(img1_val, img1_weight)
showplot(img2_val, img2_weight)
showplot(img3_val, img3_weight)
plt.legend(["Image-1","Image-2","Image-3"])