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
        >>> arr = cv2.imread("assets/img/doctests/single_line_lcd.jpg", 0)
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
    def plot_projection(projection, image, window_title="Projection"):
        projection = np.array(projection).squeeze()

        fig = plt.figure()
        plt.gcf().canvas.set_window_title(window_title)

        a = fig.add_subplot(311)
        a.set_title("Derivative")
        derivative = np.gradient(projection) / projection
        plt.plot(derivative)
        plt.plot([2.5 * np.std(derivative)] * len(derivative))
        plt.plot([-2.5 * np.std(derivative)] * len(derivative))

        a = fig.add_subplot(312)
        a.set_title("Projection")
        plt.plot(projection)

        a = fig.add_subplot(313)
        a.set_title("Image")
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

    @staticmethod
    def angles_to_points(lines):
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 100*(-b))
            y1 = int(y0 + 100*(a))
            x2 = int(x0 - 100*(-b))
            y2 = int(y0 - 100*(a))
            yield (x1, y1), (x2, y2)
