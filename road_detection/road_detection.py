import cv2
import numpy as np
from sklearn.cluster import KMeans



def gamma_correction(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)





def color_clustering(image):
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=3)
    clt.fit(image)
    print(clt.labels_, clt.labels_.__len__())
    print(np.unique(clt.labels_))


img = cv2.imread("./road_image/cctv (2).jpg", cv2.IMREAD_COLOR)
color_clustering(img)
img2 = cv2.GaussianBlur(img,(9,9),0)
img3 = cv2.bilateralFilter(img,9,75,75)


f = np.fft.fft2(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
fshift = np.fft.fftshift(f)


crow, ccol = img.shape[0]//2, img.shape[1]//2
d = 3
fshift[crow-d:crow+d, ccol-d:ccol+d] = 1



f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)


kernel = np.ones((3, 3), np.uint8)
#img_back2  = cv2.dilate(img_back.astype("uint8"), kernel, iterations=1)



for i in range(5):
    img_back = gamma_correction(img_back.astype("uint8"), 1.05)
    img_back = cv2.morphologyEx(img_back.astype("uint8"), cv2.MORPH_CLOSE, kernel)



#ret, img_back3 = cv2.threshold(img_back.astype("uint8"),10,255, cv2.THRESH_TOZERO)
#img_back3 = cv2.adaptiveThreshold(img_back.astype("uint8"),255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)
#img_back3 = gamma_correction(img_back.astype("uint8"), 1.5)



cv2.imshow("orig", img)
cv2.imshow("gaussian blurred", img2)
cv2.imshow("bilateral blurred", img3)
cv2.imshow("fourier filtering", img_back.astype('uint8'))
#cv2.imshow("fourier filtering - thresh", img_back2.astype('uint8'))


print(img_back.dtype)

cv2.waitKey(0)
