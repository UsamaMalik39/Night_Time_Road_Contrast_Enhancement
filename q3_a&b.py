# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:04:01 2020

@author: Malik Usama
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def create_plot_arrays(rows, cols,image):
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

def calculate_round_after_cdf(cdf_rec):
    normalized_final=np.array([0.0]*256)
    for i in range(0,256,1):
        normalized_final[i]=255*cdf_rec[i]
    return normalized_final
    
def create_new_image(orig,cdf_of_img,rows,col):
    for i in range(0,rows,1):
        for j in range(0,col,1):
            index=orig[i][j]
            orig[i][j]=cdf_of_img[index]
    return orig
    
    
    
        
        
        
    

img_1=cv.imread('C:/EME/7th sem/dip/assignment/#1/handout/data/hw1_dark_road_1.jpg')
img_2=cv.imread('C:/EME/7th sem/dip/assignment/#1/handout/data/hw1_dark_road_2.jpg')
img_3=cv.imread('C:/EME/7th sem/dip/assignment/#1/handout/data/hw1_dark_road_3.jpg')


img1_grs=cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
img2_grs=cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)
img3_grs=cv.cvtColor(img_3,cv.COLOR_BGR2GRAY)



count_of_weight=0
normalized_weight_img1=np.array([0.0]*256)
normalized_weight_img2=np.array([0.0]*256)
normalized_weight_img3=np.array([0.0]*256)

"size of img_1 is (450,800,3)"
img1_val,img1_weight=create_plot_arrays(450, 800,img1_grs)


"size of img_2 is (480,640,3)"
img2_val,img2_weight=create_plot_arrays(480, 640,img2_grs)


"size of img_3 is (450,800,3)"
img3_val,img3_weight=create_plot_arrays(450, 800,img3_grs)


"""showplot(img1_val,img1_weight)
showplot(img2_val,img2_weight)
showplot(img3_val,img3_weight)

plt.legend(["Image-1","Image-2","Image-3"])"""


"part b"

cdf_img1=calculate_cdf(img1_val,img1_weight,normalized_weight_img1)
cdf_img2=calculate_cdf(img2_val,img2_weight,normalized_weight_img2)
cdf_img3=calculate_cdf(img3_val,img3_weight,normalized_weight_img3)

final_array_1=calculate_round_after_cdf(cdf_img1)
final_array_2=calculate_round_after_cdf(cdf_img2)
final_array_3=calculate_round_after_cdf(cdf_img3)

"""showplot(img1_val, final_array_1)
showplot(img2_val, final_array_2)
showplot(img3_val, final_array_3)
plt.legend(["Image-1","Image-2","Image-3"])"""


remake_img1=create_new_image(img1_grs, final_array_1, 450, 800)
remake_img2=create_new_image(img2_grs, final_array_2, 480, 640)
remake_img3=create_new_image(img3_grs, final_array_3, 450, 800)

remake_img1_val,remake_img1_weight=create_plot_arrays(450, 800, remake_img1)
remake_img2_val,remake_img2_weight=create_plot_arrays(480, 640, remake_img2)
remake_img3_val,remake_img3_weight=create_plot_arrays(450, 800, remake_img3)

remake_img1_val=img1_val
remake_img2_val=img2_val
remake_img3_val=img3_val

"to see histogram of modified images grayscale values"

showplot(remake_img1_val,remake_img1_weight)
showplot(remake_img2_val,remake_img2_weight)
showplot(remake_img3_val,remake_img3_weight)
plt.title("modified images grayscale hist")
plt.legend(["Image-1","Image-2","Image-3"])

cv.imshow("remake-1",remake_img1)
cv.waitKey(0)

cv.imshow("remake-2",remake_img2)
cv.waitKey(0)

cv.imshow("remake-3",remake_img3)
cv.waitKey(0)


            
    
            
            




    