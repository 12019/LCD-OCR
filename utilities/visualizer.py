import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
from utilities.rectangle import Rectangle


class Visualizer:

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
        Visualizer.show_image(canvas, "Contours")

    @staticmethod
    def outline(ndarray, first_coord_list, second_coord_list=None, window_title="Outline"):
        """Visual representation of target areas

        :param ndarray: visible
        :param first_coord_list, second_coord_list: list of Coordinate objects
        :return:

        >>> arr = cv2.imread('assets/img/doctests/photo.jpg', 0)
        >>> r1 = Rectangle(400, 500, 1200, 300)
        >>> r2 = Rectangle(100, 400, 150, 300)
        >>> r3 = Rectangle(250, 200, 500, 100)
        >>> Visualizer.outline(arr, [r1, r2], [r3])
        """
        assert isinstance(ndarray, np.ndarray)
        for coord in first_coord_list:
            ndarray = cv2.rectangle(
                ndarray,
                (int(coord.left), int(coord.top)),
                (int(coord.left + coord.width), int(coord.top + coord.height)),
                (0, 255, 0),
                10)
        for coord in second_coord_list or []:
            ndarray = cv2.rectangle(
                ndarray,
                (int(coord.left), int(coord.top)),
                (int(coord.left + coord.width), int(coord.top + coord.height)),
                (255, 0, 255),
                10)
        Visualizer.show_image(ndarray, window_title)
