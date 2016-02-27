import numpy as np
from operator import itemgetter
import cv2

from utilities.plotter import Plotter


class Projection:
    image = None
    measurement = -1
    orientation = None

    TYPE_HORIZONTAL = 1
    TYPE_VERTICAL = 0

    projection = []

    def __init__(self, full_color_ndarray, orientation, blur_percent=0.2):
        # todo make orientation the second parameter
        assert isinstance(full_color_ndarray, np.ndarray)
        assert len(full_color_ndarray.shape) == 2, "Not a valid visible"
        self.image = Projection.preprocess_image(full_color_ndarray, blur_percent)
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

    # def make_colorful_projection(self, interpolation_type=cv2.INTER_CUBIC):
    #     """List of average pixel values along the axis
    #
    #     >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    #     >>> # horizontal
    #     >>> bins = Projection(arr, Projection.TYPE_HORIZONTAL)
    #     >>> bins.make_colorful_projection()[3]
    #     0.10964912280701754
    #     >>> # bins.debug("colorful horizontal")
    #     >>>
    #     >>> # vertical
    #     >>> bins = Projection(arr, Projection.TYPE_VERTICAL)
    #     >>> bins.make_colorful_projection()[3]
    #     0.53275109170305679
    #     >>> # bins.debug("colorful vertical")  # todo AttributeError: 'NoneType' object has no attribute 'tk'
    #     """
    #     new_shape = (1, self.measurement) if self.orientation == self.TYPE_HORIZONTAL else (self.measurement, 1)
    #     vector = cv2.resize(self.image, new_shape, interpolation=interpolation_type).flatten()
    #     vector = 255 - vector
    #     assert len(vector) == self.measurement
    #     return Projection.normalize(vector)

    def make_binary_projection(self):
        """Alternative to colorful histogram
        Count black pixels in each column

        >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
        >>> # horizontal
        >>> bins = Projection(arr, Projection.TYPE_HORIZONTAL)
        >>> bins.make_binary_projection()[3]
        0.001968503937007874
        >>> # bins.debug("binary horizontal")
        >>>
        >>> # vertical
        >>> bins = Projection(arr, Projection.TYPE_VERTICAL)
        >>> bins.make_binary_projection()[3]
        0.0011198208286674132
        >>> # bins.debug("binary vertical")
        """
        # binary = cv2.adaptiveThreshold(self.image, max_value, adaptive_method, cv2.THRESH_BINARY_INV, block_size, c)
        th, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary[binary > 0] = 1
        vector = np.sum(binary, axis=1-self.orientation)
        assert len(vector) == self.measurement
        return Projection.normalize(vector)

    def get_peaks_function(self):
        projection = self.make_binary_projection()  # todo try equalizeHist and blur
        return np.gradient(projection) / projection

    def get_peaks(self):
        """
        >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
        >>> projection = Projection(arr, Projection.TYPE_HORIZONTAL)
        >>> list(projection.get_peaks())
        [(15, 3.5), (184, 2.5), (1022, 2.5)]
        >>> # projection.debug()
        """
        peaks_function = self.get_peaks_function()
        threshold = 2.5 * np.std(peaks_function)
        current = []
        for i, value in enumerate(peaks_function):
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
        Plotter.projection(self.make_binary_projection(), binary, window_title)
