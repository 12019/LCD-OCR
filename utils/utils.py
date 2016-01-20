import math
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2


class Utils:

    def __init__(self):
        assert False, "This class is for static methods only"

    @staticmethod
    def show_image(image, window_title, image_map=None):
        if 'DISPLAY' not in os.environ:
            return
        plot = plt.imshow(image, image_map)
        plt.title(window_title)
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        plt.show()

    @staticmethod
    def show_images(images, labels, window_title):
        """
        >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg", 0)
        >>> # Utils.show_images([arr, arr], ['t', 't'], "general title")
        """
        if 'DISPLAY' not in os.environ:
            return
        fig = plt.figure()
        fig.canvas.set_window_title(window_title)
        for i, image in enumerate(images):
            label = labels[i]
            a = fig.add_subplot(len(images), 1, i+1)
            a.set_title(label)
            plt.imshow(image)
        plt.show()

    @staticmethod
    def show_contour(shape, contours, contour_id=-1):
        height, width = shape
        canvas = np.zeros((height, width, 3), np.uint8)
        cv2.drawContours(canvas, contours, contour_id, (0, 255, 0), 3)
        Utils.show_image(canvas, "Contours")

    @staticmethod
    def subimage_rotated(image, center, theta, width, height):
        theta *= 3.14159 / 180
        v_x = (math.cos(theta), math.sin(theta))
        v_y = (-math.sin(theta), math.cos(theta))
        s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
        s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

        mapping = np.array([[v_x[0], v_y[0], s_x],
                            [v_x[1], v_y[1], s_y]])

        return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
