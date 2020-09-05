import cv2
import numpy as np
from matplotlib import pyplot as plt


img1 = cv2.imread('1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = img1[:300, :300]


# img1을 shift 시켜 img2 생성
trans = np.float32([[1,0,-30],[0,1,-33]])
rows,cols = img1.shape

img2 = cv2.warpAffine(img1,trans,(cols,rows))


# 이미지의 fft 계산 후 normalize 하여 exp(j*theta) 성분만 추출
img1_f = np.fft.fft2(img1)
img1_f = img1_f/np.abs(img1_f)

img2_f = np.fft.fft2(img2)
img2_f = img2_f/np.abs(img2_f)



# 두 이미지 간 phase 차이를 구하고 inverse fft
phs_corr = img1_f/img2_f
result = np.fft.ifft2(phs_corr)
result = (np.abs(result) * 255).astype(np.uint8)

print(result)

plt.figure(figsize=(14, 6))
plt.subplot(131),plt.imshow(img1, cmap = 'gray')
plt.title('Input Image'), plt.xticks(range(0, cols, 50)), plt.yticks(range(0, cols, 50))

plt.subplot(132),plt.imshow(img2, cmap = 'gray')
plt.title('Phase shifted Image'), plt.xticks(range(0, cols, 50)), plt.yticks(range(0, cols, 50))

plt.subplot(133),plt.imshow(result, cmap = 'gray')
plt.title('Phase correlation'), plt.xticks(range(0, cols, 50)), plt.yticks(range(0, cols, 50))
plt.show()
