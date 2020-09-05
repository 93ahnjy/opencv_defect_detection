import numpy as np
import math
import os
import cv2



def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))




filelist_len = len(os.listdir('./videos_output4'))
print(filelist_len)
psnr_list = []
for i in range(1, filelist_len):


    num = str(i).zfill(3)


    pred_file    = './videos_output4/out{}.jpg'.format(num)
    gt_file      = './video4/video4_{}.jpg'.format(num)

    pred = cv2.imread(pred_file)
    gt   = cv2.imread(gt_file)

    print(i)
    psnr = PSNR(pred, gt)
    psnr_list.append(psnr)


    print('PSNR: %s dB' % psnr)



print("Mean of PSNR :", np.array(psnr_list).mean())
