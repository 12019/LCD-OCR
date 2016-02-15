# -*- coding: utf-8 -*-

# import numpy as np
import numpy as np

import cv2

from coordinates import Coordinates
from utils.utils import Utils


class Projection:
    # """
    # >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    # >>> bins = Projection(arr, 10, Projection.TYPE_HORIZONTAL)
    # >>> bins.make_colorful_projection()
    # array([ 0.        ,  0.25520833,  0.9375    ,  0.41145833,  0.65104167,
    #         0.65104167,  1.        ,  0.52083333,  0.359375  ,  0.16666667])
    # >>> bins.make_binary_projection()
    # array([ 0.18173988,  0.13436693,  0.96210164,  0.47114556,  1.        ,
    #         0.78466839,  0.99483204,  0.27820844,  0.31007752,  0.        ])
    # >>> # basis = Coordinates.from_ndarray(arr)
    # >>> # print list(bins.find_areas(basis, 127))
    # [(0:572, 0:1310)]
    # >>> # bins.debug()
    # """
    image = None
    bins_count = -1
    orientation = None

    TYPE_VERTICAL = 0
    TYPE_HORIZONTAL = 1

    projection = []

    def __init__(self, full_color_ndarray, count, orientation, blur_percent=0.2):
        # todo make orientation the second parameter
        assert isinstance(full_color_ndarray, np.ndarray)
        assert len(full_color_ndarray.shape) == 2, "Not a valid visible"
        self.image = Projection.preprocess_image(full_color_ndarray, blur_percent)
        self.bins_count = count
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

        :param interpolation_type:
        :return:
        """
        new_shape = (1, self.bins_count) if self.orientation == self.TYPE_VERTICAL else (self.bins_count, 1)
        vector = cv2.resize(self.image, new_shape, interpolation=interpolation_type).flatten()
        vector = 255 - vector
        return Projection.normalize(vector)

    def make_binary_projection(self):
        """Alternative to colorful projection
        Count black pixels in each column
        """
        # binary = cv2.adaptiveThreshold(self.image, max_value, adaptive_method, cv2.THRESH_BINARY_INV, block_size, c)
        th, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        shape = list(binary.shape)
        if self.orientation == self.TYPE_VERTICAL:
            shape[1] = self.bins_count
        else:
            shape[0] = self.bins_count
        binary = cv2.resize(binary, tuple(shape), cv2.INTER_CUBIC)
        # Utils.show_image(binary, "binary")
        binary[binary > 0] = 1
        vector = np.sum(binary, axis=0)
        return Projection.normalize(vector)

    @staticmethod
    def normalize(arr):
        arr = 1.0 * arr - min(arr)
        arr[:] += 1
        return arr / max(arr)

    def debug(self, window_title="Color Bins"):
        # colored, monochrome and vector
        # assert len(self.projection), "Please make a projection"
        # assert self.binary_projection.shape, "Make binary projection or refactor debug code"
        # linspace = np.linspace(0, 255, 255)
        # reference_colors = [linspace, linspace, linspace]
        # histogram_label = ["Vertical projection", "Horizontal projection"][self.orientation]
        # print self.image.shape
        # Utils.show_images([self.image, self.colorful_projection, self.binary_projection, reference_colors],
        #                   ["Full color", histogram_label, "Binary projection", u"Threshold = %d ⋲ 0..255" % self.thresh],
        #                   window_title)
        # Utils.plot_projection(self.make_colorful_projection(), self.image)
        Utils.plot_projection(self.make_binary_projection(), self.image)


class AreaFactory:
    """
    >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> f = AreaFactory(arr, 100, Coordinates.from_ndarray(arr), Projection.TYPE_VERTICAL)
    >>> list(f.find_areas())
    [(39:463, 0:1310)]
    # >>> f = AreaFactory(arr, 100, Coordinates.from_ndarray(arr), Projection.TYPE_HORIZONTAL)
    # >>> list(f.find_areas())
    # [(39:463, 0:1310)]
    """
    projection = None
    basis = None
    bins_count = 10
    compression = -1
    orientation = Projection.TYPE_VERTICAL

    def __init__(self, full_color_image, bins_count, basis, orientation):
        self.projection = Projection(full_color_image, bins_count, orientation, 0.2).make_binary_projection()
        Utils.plot_projection(self.projection, full_color_image)
        self.projection = AreaFactory.binarize_projection(self.projection)
        # print self.projection
        self.basis = basis
        measurement = full_color_image.shape[0] if orientation == Projection.TYPE_VERTICAL else full_color_image.shape[1]
        self.compression = 1.0 * measurement / bins_count
        self.bins_count = bins_count

    def find_areas(self, overflow_percent=0.03):  # todo gradients
        """
        :param overflow_percent:    0 to 1
        :return:     [(10, 100), (150, 300)]
        """
        r = self._split_projection_into_ranges()
        s = self._scale_ranges(r, overflow_percent)
        return self._convert_ranges_into_coordinates(self.basis, s)

    def _split_projection_into_ranges(self):
        start = False
        stop = False
        is_inside = False
        for index, i in enumerate(np.round(self.projection)):
            if i and not is_inside:
                is_inside = True
                start = index
            elif not i and is_inside:
                is_inside = False
                stop = index
                yield (start, stop)
        if not stop:
            yield (start, self.bins_count)

    def _scale_ranges(self, ranges, overflow_size):
        for left, right in ranges:
            # print 'scale', region, bins_count, offset
            left_scaled = max(0, (left - overflow_size) * self.compression)
            right_scaled = min(self.bins_count, right + overflow_size) * self.compression
            # print 'scaled:', left, right
            yield left_scaled, right_scaled

    def _convert_ranges_into_coordinates(self, basis, ranges):
        for left_scaled, right_scaled in ranges:
            if self.orientation is Projection.TYPE_VERTICAL:
                yield Coordinates(basis.top + left_scaled, basis.top + right_scaled, basis.left, basis.right)
            else:
                yield Coordinates(basis.top, basis.bottom, basis.left + left_scaled, basis.left + right_scaled)

    @staticmethod
    def binarize_projection(projection):
        return projection > np.average(projection)
