import numpy as np
from operator import itemgetter
import cv2

from utilities.visualizer import Visualizer


class Projection:
    image = None
    measurement = -1
    orientation = None

    TYPE_VERTICAL = 1
    TYPE_HORIZONTAL = 0

    projection = []

    def __init__(self, full_color_ndarray, orientation, blur_percent=0.2):
        # todo make orientation the second parameter
        assert isinstance(full_color_ndarray, np.ndarray)
        assert len(full_color_ndarray.shape) == 2, "Not a valid visible"
        self.image = Projection.preprocess_image(full_color_ndarray, blur_percent)
        self.measurement = full_color_ndarray.shape[orientation]
        self.orientation = orientation

    @staticmethod
    def preprocess_image(full_color_ndarray, blur_percent):
        blur_kernel = Projection.odd(full_color_ndarray.shape[0] * blur_percent)
        img = cv2.equalizeHist(full_color_ndarray)
        return cv2.medianBlur(img, blur_kernel)

    @staticmethod
    def odd(number):
        number = int(number)
        if not number % 2:
            number += 1
        return number

    def make_colorful_projection(self, interpolation_type=cv2.INTER_CUBIC):
        """List of average pixel values along the axis

        >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
        >>> # horizontal
        >>> bins = Projection(arr, 100, Projection.TYPE_HORIZONTAL)
        >>> bins.make_colorful_projection()[3]
        0.10964912280701754
        >>> # bins.debug("colorful horizontal")
        >>>
        >>> # vertical
        >>> bins = Projection(arr, 100, Projection.TYPE_VERTICAL)
        >>> bins.make_colorful_projection()[3]
        0.53275109170305679
        >>> # bins.debug("colorful vertical")  # todo AttributeError: 'NoneType' object has no attribute 'tk'
        """
        new_shape = (1, self.measurement) if self.orientation == self.TYPE_VERTICAL else (self.measurement, 1)
        vector = cv2.resize(self.image, new_shape, interpolation=interpolation_type).flatten()
        vector = 255 - vector
        assert len(vector) == self.measurement
        return Projection.normalize(vector)

    def make_binary_projection(self):
        """Alternative to colorful histogram
        Count black pixels in each column

        >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
        >>> # horizontal
        >>> bins = Projection(arr, 100, Projection.TYPE_HORIZONTAL)
        >>> bins.make_binary_projection()[3]
        0.13058419243986255
        >>> # bins.debug("binary horizontal")
        >>>
        >>> # vertical
        >>> bins = Projection(arr, 100, Projection.TYPE_VERTICAL)
        >>> bins.make_binary_projection()[3]
        0.12531969309462915
        >>> # bins.debug("binary vertical")
        """
        # binary = cv2.adaptiveThreshold(self.image, max_value, adaptive_method, cv2.THRESH_BINARY_INV, block_size, c)
        th, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        shape = list(binary.shape)
        shape[self.orientation] = self.measurement
        binary = cv2.resize(binary, tuple(shape), cv2.INTER_CUBIC)
        binary[binary > 0] = 1
        vector = np.sum(binary, axis=self.orientation)
        assert len(vector) == self.measurement
        return Projection.normalize(vector)

    def get_peaks(self):
        """
        >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
        >>> projection = Projection(arr, 100, Projection.TYPE_HORIZONTAL, 0.2)
        >>> list(projection.get_peaks())
        [(0, 55.000000000000007), (13, 41.5), (77, 82.5)]
        >>> # projection.debug()
        """
        projection = self.make_binary_projection()
        derivative = np.gradient(projection) / projection
        threshold = 2.5 * np.std(derivative)
        current = []
        for i, value in enumerate(derivative):
            if value > threshold:
                current.append((i, value))
            elif len(current):
                yield max(current, key=itemgetter(1))
                current = []

    @staticmethod
    def normalize(arr):
        arr = 1.0 * arr - min(arr)
        arr[:] += 1
        return arr / max(arr)

    def debug(self, window_title="Projection"):
        th, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        Visualizer.plot_projection(self.make_binary_projection(), binary, window_title)
