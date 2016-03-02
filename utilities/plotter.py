import numpy as np
from matplotlib import pyplot as plt
import os
import cv2

from utilities.rectangle import Rectangle


class Plotter:

    def __init__(self):
        assert False, "This class made exclusively for static methods"

    @staticmethod
    def image(image, window_title="Plotter", image_map=None):
        if 'DISPLAY' not in os.environ:
            return
        plot = plt.imshow(image, image_map)
        plt.title(window_title)
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        plt.show()

    @staticmethod
    def images(images, labels, window_title="Plotter"):
        """
        >>> arr = cv2.imread("assets/img/doctests/single_line_lcd.jpg", 0)
        >>> # Visualizer.show_images([arr, arr], ['t', 't'], "general title")
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
    def projection(projection, image, window_title="Projection"):
        if 'DISPLAY' not in os.environ:
            return
        projection = np.array(projection).squeeze()

        fig = plt.figure()
        plt.gcf().canvas.set_window_title(window_title)

        a = fig.add_subplot(311)
        a.set_title("Derivative")
        derivative = np.gradient(projection)  # / projection
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
    def contour(shape, contours, contour_id=-1):
        height, width = shape
        canvas = np.zeros((height, width, 3), np.uint8)
        cv2.drawContours(canvas, contours, contour_id, (0, 255, 0), 3)
        Plotter.image(canvas, "Contours")

    @staticmethod
    def outline_rectangles(ndarray, first_rect_list, second_rect_list=None, window_title="Outline", projection=None):
        """Visual representation of target areas

        :param ndarray: visible
        :param first_coord_list, second_coord_list: list of Coordinate objects
        :return:

        >>> arr = cv2.imread('assets/img/doctests/multi_line_lcd.jpg', 0)  # 384x344
        >>> r1 = Rectangle(100, 100, 100, 100)
        >>> r2 = Rectangle(200, 100, 200, 100)
        >>> r3 = Rectangle(0, 100, 200, 100)
        >>> # Plotter.outline(arr, [r1, r2], [r3])
        """
        assert isinstance(ndarray, np.ndarray)
        ndarray = ndarray.copy()  # http://stackoverflow.com/questions/23830618/
        for coord in first_rect_list:
            ndarray = cv2.rectangle(
                ndarray,
                (int(coord.left), int(coord.top)),
                (int(coord.left + coord.width), int(coord.top + coord.height)),
                (0, 255, 0),
                10)
        for coord in second_rect_list or []:
            ndarray = cv2.rectangle(
                ndarray,
                (int(coord.left), int(coord.top)),
                (int(coord.left + coord.width), int(coord.top + coord.height)),
                (255, 0, 255),
                10)
        image_title = "%s %s %s" % (window_title, list(first_rect_list), list(second_rect_list or []) or "")
        if projection is not None:
            Plotter.projection(projection, ndarray, image_title)
        else:
            Plotter.image(ndarray, image_title)

    @staticmethod
    def outline_subimages(main_image, first_subimages_list, second_subimages_list=None, window_title="Outline"):
        assert not isinstance(main_image, np.ndarray)
        assert isinstance(first_subimages_list, list)
        assert not isinstance(first_subimages_list[0], np.ndarray)
        ndarray = main_image.ndarray
        first_rect_list = [ci.coord for ci in first_subimages_list]
        second_rect_list = [ci.coord for ci in second_subimages_list or []]
        Plotter.outline_rectangles(ndarray, first_rect_list, second_rect_list, window_title, main_image.latest_projection)
