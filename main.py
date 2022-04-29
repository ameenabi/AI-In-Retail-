import cv2, os
import numpy as np
from matplotlib import pyplot as plt

path = "./inputs/"
files = os.listdir(path)

for idx in range(len(files)):

    org = cv2.imread(path + files[idx])
    org_img = org.copy()

    img = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    SE_line1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    SE_line2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            
    img =th2.copy()
    edges = cv2.Canny(img,100,200)

    edges = cv2.dilate(edges,SE_line1,iterations = 1) 
    edges = cv2.dilate(edges,SE_line2,iterations = 1) 

    edges = cv2.erode(edges,SE_line1,iterations = 1) 
    edges = cv2.erode(edges,SE_line2,iterations = 1) 

    edges = cv2.bitwise_not(edges)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(org, contours, -1, (255,255,255), 15)
    
    th2 = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    ret2,th2 = cv2.threshold(th2,0,255,cv2.THRESH_OTSU)
    
    th2 = cv2.bitwise_not(th2)
    contours, _ = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        
        if area > 5000:
            cv2.drawContours(org_img, contours, i, (0,255,0), 4)
    
    cv2.imwrite("./outputs/"+files[idx], org_img)
