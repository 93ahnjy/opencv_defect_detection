import cv2
import numpy as np




orig_img = cv2.imread('water.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(orig_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
img = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT)


#img = cv2.GaussianBlur(img, (3,3),0)
inv_img = 255 - img






circles = cv2.HoughCircles(image=img,
                           method=cv2.HOUGH_GRADIENT,
                           dp=1,
                           minDist=10,
                           param1=50,param2=25,minRadius=10, maxRadius=80)

circles = np.uint16(np.around(circles))


#img = cv2.bitwise_and(img, mask)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)


h, w, c = cimg.shape
for i in circles[0,:]:


        cv2.circle(cimg,(i[0],i[1]),(i[2]),(255,0,0),2)
        cv2.circle(cimg,(i[0],i[1]),1,(0,0,255),3)


cimg = cv2.resize(cimg[10:-10,10:-10, :], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

cv2.imshow('water', cimg)
cv2.imwrite('detect.png', cimg)
cv2.waitKey(5000)