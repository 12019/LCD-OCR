import numpy as np
from operator import itemgetter
import cv2

from utilities.plotter import Plotter


class Projection:
    image = latest_reference_image = None
    measurement = -1
    orientation = None

    TYPE_HORIZONTAL = 1
    TYPE_VERTICAL = 0

    projection = []

    def __init__(self, full_color_ndarray, orientation, blur_percent=0.2):
        # todo make orientation the second parameter
        assert isinstance(full_color_ndarray, np.ndarray)
        assert len(full_color_ndarray.shape) == 2, "Not a valid visible"
        self.image = self.latest_reference_image = Projection.preprocess_image(full_color_ndarray, blur_percent)
        self.orientation = orientation
        self.measurement = full_color_ndarray.shape[orientation]

    @staticmethod
    def preprocess_image(full_color_ndarray, blur_percent):
        blur_kernel = Projection.to_odd(full_color_ndarray.shape[0] * blur_percent)
        img = cv2.equalizeHist(full_color_ndarray)
        return cv2.medianBlur(img, blur_kernel)

    @staticmethod
    def to_odd(number):
        number = int(number)
        if not number % 2:
            number += 1
        return number

    def get_peaks_function(self):
        projection = self.get_projection()  # todo try equalizeHist and blur
        return np.gradient(projection) / projection

    def get_peaks(self):
        # """
        # >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
        # >>> projection = BinaryProjection(arr, Projection.TYPE_HORIZONTAL)
        # >>> list(projection.get_peaks())
        # [(15, 3.5), (184, 2.5), (1022, 2.5)]
        # >>> # projection.debug()
        # """
        peaks_function = self.get_peaks_function()
        threshold = 2.5 * np.std(peaks_function)
        current = []
        for i, value in enumerate(peaks_function):
            if value > threshold:
                current.append((i, value))
            elif len(current):
                yield max(current, key=itemgetter(1))
                current = []

    def get_projection(self):
        raise NotImplementedError

    @staticmethod
    def normalize(arr):
        arr = 1.0 * arr - min(arr)
        arr[:] += 1
        return arr / max(arr)

    def debug(self, window_title="Projection"):
        Plotter.projection(self.get_projection(), self.latest_reference_image, window_title)
