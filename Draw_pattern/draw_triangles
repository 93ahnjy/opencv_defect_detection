import cv2
from math import cos, sin, radians,sqrt
import numpy as np



def init_triangle(length, center):

    dist = length * 2/sqrt(3)
    y_cen, x_cen = center

    # p1 = (int(y_cen - dist),     int(x_cen))
    # p2 = (int(y_cen + 0.5*dist), int(x_cen - sqrt(3)/2 * dist))
    # p3 = (int(y_cen + 0.5*dist), int(x_cen + sqrt(3)/2 * dist))

    p1 = (int(x_cen), int(y_cen - dist),)
    p2 = (int(x_cen - sqrt(3)/2 * dist), int(y_cen + 0.5*dist))
    p3 = (int(x_cen + sqrt(3)/2 * dist), int(y_cen + 0.5*dist))

    return [p1, p2, p3]
















def Rotated_points(points, degree, scale, center):
    x_cen, y_cen = center

    p_new = []
    for i in range(len(points)):
        x, y = points[i]
        x_ = int(scale * ((x-x_cen)*cos(radians(degree)) - (y-y_cen)*sin(radians(degree))) + x_cen)
        y_ = int(scale * ((x-x_cen)*sin(radians(degree)) + (y-y_cen)*cos(radians(degree))) + y_cen)

        p_new.append((x_, y_))

    return p_new





def draw_lines(img, points):

    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        img = cv2.line(img, p1, p2, 255, 1)

    return img





if __name__ == '__main__':

    img = np.zeros((512, 512))
    points = init_triangle(100, (256, 256))




    all_points = []
    all_points += points
    iter_num = 50

    for _ in range(iter_num):
        points = Rotated_points(points, degree=-3, scale=1.05, center=(256,256))
        all_points += points
        img = draw_lines(img, all_points)


    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
