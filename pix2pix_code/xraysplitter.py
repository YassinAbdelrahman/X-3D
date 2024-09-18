Xray = "Data/X-rays/XrayANON5UB95J1SC.png"

image = "XrayANON5UB95J1SC"

import cv2 # type: ignore
 
img = cv2.imread(Xray)
 
# cv2.imread() -> takes an image as an input
h, w, channels = img.shape
 
half = w//4
 
 
# this will be the first column
# left_part = img[:, half:half*2] 
# # this will be the second column
# right_part = img[:, half*2:half*3]  

full = img[:, half:half*3]
 
# [:,:half] means all the rows and
# all the columns upto index half
 

 
# [:,half:] means all the rows and all
# the columns from index half to the end
# cv2.imshow is used for displaying the image
# cv2.imshow('Left part', left_part)
# cv2.imshow('Right part', right_part)
 
# # this is horizontal division
# half2 = h//2
 
# top = img[:half2, :]
# bottom = img[half2:, :]
 
# cv2.imshow('Top', top)
# cv2.imshow('Bottom', bottom)
 
# saving all the images
# cv2.imwrite() function will save the image 
# into your pc
# cv2.imwrite('Data/test/xraysplit/top.jpg', top)
# cv2.imwrite('Data/test/xraysplit/bottom.jpg', bottom)
# cv2.imwrite(f'Data/test/xraysplit/{image}_right.png', right_part)
# cv2.imwrite(f'Data/test/xraysplit/{image}_left.png', left_part)
# cv2.imwrite(f'Data/test/xraysplit/{image}_right.png', right_part)

cv2.imwrite(f'Data/test/xraysplit/{image}_full.png', full)

cv2.waitKey(0)