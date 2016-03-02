import numpy as np
import cv2

from projection.abstract import Projection
from utilities.plotter import Plotter


class BinaryProjection(Projection):
    """Apply cv2.threshold, count black pixels in each column

    >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> # horizontal
    >>> bins = BinaryProjection(arr, Projection.TYPE_HORIZONTAL)
    >>> bins.get_projection()[3]
    0.001968503937007874
    >>> bins.debug("binary horizontal")
    >>>
    >>> # vertical
    >>> bins = BinaryProjection(arr, Projection.TYPE_VERTICAL)
    >>> bins.get_projection()[3]
    0.0011198208286674132
    >>> # bins.debug("binary vertical")
    """

    # def get_morph_kernel(self):
    #     kernel_size = self.get_stroke_width(1)  # todo rows count
    #     return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def get_projection(self):
        th, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.latest_reference_image = binary
        # after = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.get_morph_kernel() * 20)
        # Plotter.images([binary, after], ["before", "after"])
        binary[binary > 0] = 1
        vector = np.sum(binary, axis=1-self.orientation)
        assert len(vector) == self.measurement
        return self.normalize(vector)
