# -*- coding: utf-8 -*-

# import numpy as np
import numpy as np

import cv2

from coordinates import Coordinates
from utils.utils import Utils


class Projection:
    """
    >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg", 0)
    >>> bins = Projection(arr, 10, Projection.TYPE_HORIZONTAL)
    >>> bins.make_binary_projection()
    >>> basis = Coordinates.from_ndarray(arr)
    >>> print list(bins.find_areas(basis, 127))
    [(0:572, 0:1310)]
    >>> # bins.debug()
    """
    image = None
    bins_count = -1
    orientation = None
    compression = -1
    thresh = 0

    TYPE_VERTICAL = 0
    TYPE_HORIZONTAL = 1

    colorful_projection = binary_projection = []

    def __init__(self, full_color_ndarray, count, orientation, interpolation_type=cv2.INTER_AREA):
        assert isinstance(full_color_ndarray, np.ndarray)
        assert len(full_color_ndarray.shape) == 2, "Not a valid visible"
        self.image = full_color_ndarray
        self.bins_count = count
        self.orientation = orientation
        self.make_colorful_projection(interpolation_type)
        measurement = self.image.shape[0] if orientation == self.TYPE_VERTICAL else self.image.shape[1]
        self.compression = 1.0 * measurement / self.bins_count

    def make_colorful_projection(self, interpolation_type=cv2.INTER_CUBIC):
        new_shape = (1, self.bins_count) if self.orientation == self.TYPE_VERTICAL else (self.bins_count, 1)
        vector = cv2.resize(self.image, new_shape, interpolation=interpolation_type)
        self.colorful_projection = np.transpose(vector) if self.orientation == self.TYPE_VERTICAL else vector

    def make_binary_projection(self, thresh=127, desaturation_type=cv2.THRESH_BINARY + cv2.THRESH_OTSU, maxval=255):
        assert self.colorful_projection.shape, "Make color projection first"
        self.thresh = thresh
        th, self.binary_projection = cv2.threshold(self.colorful_projection, self.thresh, maxval, desaturation_type)
        self.binary_projection[self.binary_projection == 0] = 1
        self.binary_projection[self.binary_projection == 255] = 0

    def _split_projection_into_ranges(self):
        assert self.binary_projection.shape, "Set other parameters before calling this method"
        start = False
        stop = False
        is_inside = False
        for index, i in enumerate(self.binary_projection[0]):
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
            if self.orientation is self.TYPE_VERTICAL:
                yield Coordinates(basis.top + left_scaled, basis.top + right_scaled, basis.left, basis.right)
            else:
                yield Coordinates(basis.top, basis.bottom, basis.left + left_scaled, basis.left + right_scaled)

    def find_areas(self, basis, overflow_percent=0.03):
        """
        :param basis:   an instance of Coordinates
        :param overflow_percent:    0 to 1
        :return:     [(10, 100), (150, 300)]
        """
        assert self.binary_projection.shape
        r = self._split_projection_into_ranges()
        s = self._scale_ranges(r, overflow_percent)
        return self._convert_ranges_into_coordinates(basis, s)

    def debug(self, window_title="Color Bins"):
        # colored, monochrome and vector
        assert len(self.colorful_projection), "Run other methods first"
        # assert self.binary_projection.shape, "Make binary projection or refactor debug code"
        # linspace = np.linspace(0, 255, 255)
        # reference_colors = [linspace, linspace, linspace]
        # histogram_label = ["Vertical projection", "Horizontal projection"][self.orientation]
        # print self.image.shape
        # Utils.show_images([self.image, self.colorful_projection, self.binary_projection, reference_colors],
        #                   ["Full color", histogram_label, "Binary projection", u"Threshold = %d ⋲ 0..255" % self.thresh],
        #                   window_title)
        Utils.plot_projection(self.colorful_projection, self.image)
