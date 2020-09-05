import cv2
import numpy as np



def edge_detector_median(img):
    dilated_img = cv2.dilate(img, np.ones((4, 4), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 11)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    return diff_img



def circle_stats(img):
    circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=80, minRadius=3, maxRadius=800)
    rads = []
    for i in circles[0, :]:
        rad = i[2]
        rads.append(rad)

    r_mean = np.array(rads).mean()
    r_std = np.array(rads).std()

    return int(round(r_mean)), int(round(r_std))




img  = cv2.imread('sample_image2.png', cv2.IMREAD_GRAYSCALE)
h,w = img.shape
mask_circle = np.zeros_like(img, np.uint8)
mask_defect = np.zeros((h,w,3), np.uint8)


img2 = cv2.bilateralFilter(img,9,20,20)
img3 = edge_detector_median(img2)
img4 = cv2.threshold(255 - img3,30,255, cv2.THRESH_TOZERO)[1]


r_mean, r_std = circle_stats(img4)
circles_big   = cv2.HoughCircles(image=img4, method=cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1=50,param2=30, minRadius=r_mean, maxRadius=r_mean + 2*r_std)


cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

for i in circles_big[0,:]:
    center = (i[0],i[1])
    rad    = i[2]
    cv2.circle(cimg, center, rad, (255,0,0), 5)
    cv2.circle(cimg, center, 1, (0, 0, 255), 3)
    cv2.circle(mask_circle, center, rad, 255, -1)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
mask_circle = 255 - cv2.dilate(mask_circle, kernel)
img5 = cv2.bitwise_and(img4, mask_circle)




contours, _ = cv2.findContours(img5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
mask_defect = cv2.drawContours(mask_defect, contours, -1, (0,0,255), 20)
result = cv2.addWeighted(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR), 0.7, mask_defect, 0.3, 0)



print(r_mean, r_std)
cv2.imwrite('1. bilat img.jpg', img2)
cv2.imwrite('2. img_edge_median.jpg', img4)
cv2.imwrite('3. circle_detect.jpg', cimg)
cv2.imwrite('4. mask_circle.jpg', mask_circle)
cv2.imwrite('5. masked_edge.jpg', img5)
cv2.imwrite('6. result.jpg', result)













# circles_small = cv2.HoughCircles(image=img4, method=cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1=50,param2=30, minRadius=r_mean - 2*r_std, maxRadius=r_mean)
# for i in circles_small[0,:]:
#     center = (i[0],i[1])
#     rad    = i[2]
#     cv2.circle(cimg,center,1,  (0,0,255), 3)
