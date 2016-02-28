import cv2
import numpy as np

from projection.abstract import Projection


class AdaptiveProjection(Projection):
    """Apply cv2.adaptiveThreshold, count black pixels in each column

    >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> # horizontal
    >>> bins = AdaptiveProjection(arr, Projection.TYPE_HORIZONTAL)
    >>> bins.get_projection()[3]
    0.0019342359767891683
    >>> # bins.debug("adaptive horizontal")
    >>>
    >>> # vertical
    >>> bins = AdaptiveProjection(arr, Projection.TYPE_VERTICAL)
    >>> bins.get_projection()[3]
    0.00097560975609756097
    >>> # bins.debug("adaptive vertical")
    """

    def get_projection(self, rows_expected=1):
        letter_height = self.image.shape[0] * 0.95 / rows_expected
        stroke_width = self.to_odd(letter_height / 10 * 2)
        binary = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, stroke_width, 5)
        # th, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.latest_reference_image = binary
        binary[binary > 0] = 1
        vector = np.sum(binary, axis=1-self.orientation)
        assert len(vector) == self.measurement
        return Projection.normalize(vector)
