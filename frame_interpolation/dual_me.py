import cv2
import pathlib
import numpy as np
import itertools
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import sub_block, validate_block


def PSNR(img1, img2):

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def show_quiver(x_component_arrows, y_components_arrows):
    plt.quiver(x_component_arrows, y_components_arrows)
    plt.show()




class DualME(object):
    def __init__(self, previous, current, N=8, R=None, delta=0.3):
        """
        :param previous: The previous frame
        :param current: The current frame
        :param N: The block size
        :param R: The search range
        :param delta: The ratio between SBOAD and SUOAD
        """
        self.previous = previous
        self.current = current
        if previous.shape != current.shape:
            raise IOError('The previous and current frames have not the same shape.')
        self.height, self.width = previous.shape[:2]
        self.height, self.width = (self.height // N * N, self.width // N * N)
        self.previous = cv2.resize(self.previous, (self.width, self.height))
        self.current = cv2.resize(self.current, (self.width, self.height))

        self.N = N
        self.R = R if (R is not None) and (type(R) is int) else N
        self.delta = delta

    def evaluate_mv(self):
        predicted_frame = np.zeros(self.previous.shape, np.uint8)
        motion_field = np.zeros((self.height // self.N, self.width // self.N, 2))
        validity_matrix = np.zeros((self.height // self.N, self.width // self.N))

        for block_idx, block_jdx in tqdm(list(itertools.product(range(0, self.height - self.N + 1, self.N),
                                                                range(0, self.width - self.N + 1, self.N)))):
            min_diff = np.infty
            matching_block = np.zeros((self.N, self.N), np.uint8)
            dx, dy = (0, 0)
            for vy, vx in itertools.product(range(-self.R, self.R + int(self.N) + 1),
                                            range(-self.R, self.R + int(self.N) + 1)):
                coord_previous_bidir_block = ((int(block_idx - vy), int(block_jdx - vx)),
                                              (int(block_idx - vy + self.N), int(block_jdx - vx + self.N)))
                coord_current_bidir_block = ((int(block_idx + vy), int(block_jdx + vx)),
                                             (int(block_idx + vy + self.N), int(block_jdx + vx + self.N)))
                coord_previous_unidir_block = ((int(block_idx - 2 * vy), int(block_jdx - 2 * vx)),
                                               (int(block_idx - 2 * vy + self.N), int(block_jdx - 2 * vx + self.N)))
                coord_current_unidir_block = ((int(block_idx), int(block_jdx)),
                                              (int(block_idx +  self.N), int(block_jdx + self.N)))

                if coord_previous_bidir_block[0][0] < 0 or coord_previous_bidir_block[0][1] < 0 or \
                        coord_previous_bidir_block[1][0] > self.height or coord_previous_bidir_block[1][1] > self.width:
                    continue
                if coord_current_bidir_block[0][0] < 0 or coord_current_bidir_block[0][1] < 0 or \
                        coord_current_bidir_block[1][0] > self.height or coord_current_bidir_block[1][1] > self.width:
                    continue
                if coord_previous_unidir_block[0][0] < 0 or coord_previous_unidir_block[0][1] < 0 or \
                        coord_previous_unidir_block[1][0] > self.height or coord_previous_unidir_block[1][1] > self.width:
                    continue
                if coord_current_unidir_block[0][0] < 0 or coord_current_unidir_block[0][1] < 0 or \
                        coord_current_unidir_block[1][0] > self.height or coord_current_unidir_block[1][1] > self.width:
                    continue

                SUOAD = np.sum(np.abs(sub_block(self.previous, coord_previous_unidir_block[0], coord_previous_unidir_block[1]).astype(np.float32) -
                                      sub_block(self.current, coord_current_unidir_block[0], coord_current_unidir_block[1]).astype(np.float32)))
                SBOAD = np.sum(np.abs(sub_block(self.previous, coord_previous_bidir_block[0], coord_previous_bidir_block[1]).astype(np.float32) -
                                      sub_block(self.current, coord_current_bidir_block[0], coord_current_bidir_block[1]).astype(np.float32)))
                SDOAD = SBOAD + SUOAD
                if SBOAD < 1e-10 or SUOAD < 1e-10:
                    validity_matrix[block_idx // self.N, block_jdx // self.N] = 1 if abs(SBOAD - SUOAD) < self.delta * ((1.5 * self.N) ** 2) else 0
                else:
                    validity_matrix[block_idx // self.N, block_jdx // self.N] = 1 if 1 - self.delta < SBOAD / SUOAD < 1 + self.delta else 0
                if SDOAD < min_diff:
                    min_diff = SDOAD
                    matching_block = sub_block(self.previous,
                                               (block_idx - vy, block_jdx - vx),
                                               (block_idx - vy + self.N, block_jdx - vx + self.N))
                    dx, dy = (vx, vy)

            predicted_frame[block_idx:block_idx+self.N, block_jdx:block_jdx+self.N] = matching_block

            motion_field[block_idx // self.N, block_jdx // self.N, 0] = dx
            motion_field[block_idx // self.N, block_jdx // self.N, 1] = dy

        return predicted_frame, motion_field, validity_matrix


class BiME(object):
    def __init__(self, previous, current, N=8, R=None):
        """
        :param previous: The previous frame
        :param current: The current frame
        :param N: The block size
        :param R: The search range
        """
        self.previous = previous
        self.current = current
        if previous.shape != current.shape:
            raise IOError('The previous and current frames have not the same shape.')
        self.height, self.width = previous.shape[:2]
        self.height, self.width = (self.height // N * N, self.width // N * N)
        self.previous = cv2.resize(self.previous, (self.width, self.height))
        self.current = cv2.resize(self.current, (self.width, self.height))

        self.N = N
        self.R = R if (R is not None) and (type(R) is int) else N

    def evaluate_mv(self):
        predicted_frame = np.zeros(self.previous.shape, np.uint8)
        motion_field = np.zeros((self.height // self.N, self.width // self.N, 2))

        for block_idx, block_jdx in tqdm(list(itertools.product(range(0, self.height - self.N + 1, self.N),
                                                                range(0, self.width - self.N + 1, self.N)))):
            min_diff = np.infty
            matching_block = np.zeros((self.N, self.N), np.uint8)
            dx, dy = (0, 0)
            for vy, vx in itertools.product(range(-self.R, self.R + self.N + 1),
                                            range(-self.R, self.R + self.N + 1)):
                valid, start_current_block, end_current_block, \
                start_previous_block, end_previous_block = validate_block(block_idx, block_jdx, vy, vx, self.height, self.width, self.N)
                if valid is False:
                    continue

                current_block = sub_block(self.current, start_current_block, end_current_block).astype(np.float32)
                previous_block = sub_block(self.previous, start_previous_block, end_previous_block).astype(np.float32)

                block_difference = np.sum(np.abs(current_block - previous_block))
                if block_difference < min_diff:
                    min_diff = block_difference
                    matching_block = previous_block
                    dx, dy = (vx, vy)

            predicted_frame[block_idx:block_idx+self.N, block_jdx:block_jdx+self.N] = matching_block
            motion_field[block_idx // self.N, block_jdx // self.N, 0] = dx
            motion_field[block_idx // self.N, block_jdx // self.N, 1] = dy

        return predicted_frame, motion_field




if __name__ == "__main__":
    img_list = sorted(os.listdir('./images/video5'))
    print(img_list)

    psnr_list = []
    for idx in range(10, 30):

        prev_img = os.path.join('./images/video5', img_list[idx])
        gt_img   = os.path.join('./images/video5', img_list[idx+1])
        curr_img = os.path.join('./images/video5', img_list[idx+2])

        previous_frame = cv2.imread(prev_img)
        ground_truth = cv2.imread(gt_img)
        current_frame = cv2.imread(curr_img)

        searcher = DualME(previous_frame, current_frame, 8, R=8)
        predicted_frame, motion_field, validity_matrix = searcher.evaluate_mv()

        #cv2.imshow('previous', searcher.previous)
        #cv2.imshow('current', searcher.current)


        #cv2.imshow('ground_truth', cv2.resize(ground_truth, dsize=((ground_truth.shape[1] // 8) * 8, (ground_truth.shape[0] // 8) * 8)))
        #cv2.imshow('predicted_frame', predicted_frame)
        motion_field_x = motion_field[:, :, 0]
        motion_field_y = motion_field[:, :, 1]
        plt.savefig(os.path.join('./images/video5_out', 'vector5_out{}.jpg'.format(str(idx+1).zfill(3))), dpi=300, bbox_inches='tight')
        show_quiver(motion_field_x, motion_field_y[::-1])


        psnr = PSNR(predicted_frame, cv2.resize(ground_truth, dsize=((ground_truth.shape[1] // 8) * 8, (ground_truth.shape[0] // 8) * 8)))
        psnr_list.append(psnr)

        print('PSNR: %s dB' % psnr)

        #cv2.waitKey(1)
        filename = os.path.join('./images/video5_out', 'video5_out{}.jpg'.format(str(idx+1).zfill(3)))
        print(filename)
        cv2.imwrite(filename, predicted_frame)
        cv2.destroyAllWindows()

    print("Mean of PSNR :", np.array(psnr_list).mean())
