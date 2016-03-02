import numpy as np
import cv2

from italics.shear_tool import ShearTool
from projection.abstract import Projection
from projection.binary import BinaryProjection
from utilities.plotter import Plotter


class FontNormalizer:
    """
    >>> before = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> obj = FontNormalizer(before)
    >>> incl = obj.find_inclination()
    >>> incl  # todo it should be closer to 8.0
    2.0
    >>> first = obj.normalize_font(incl)
    >>> # Plotter.image(first, "normalized image")
    >>> obj2 = FontNormalizer(first)
    >>> obj2.find_inclination()
    0.0
    """
    img = None

    def __init__(self, img):
        assert isinstance(img, np.ndarray)
        self.img = img

    def calculate_weights(self):
        for angle in np.arange(-16., 16., 2.):  # PAMI96, page 9
            yield angle, self.calculate_single_weight(angle)

    def calculate_single_weight(self, angle):
        """Finds the strongest gradient in the image"""
        obj = ShearTool(self.img)
        corrected = obj.shear(angle)
        p = BinaryProjection(corrected, Projection.TYPE_HORIZONTAL)
        # p.debug()
        return max(np.gradient(p.get_projection())) + abs(min(np.gradient(p.get_projection())))

    def find_inclination(self):
        weights = self.calculate_weights()
        weights = list(weights)
        val, idx = max((abs(val), idx) for (idx, val) in weights)
        # print val, idx
        return idx

    def normalize_font(self, angle):
        obj = ShearTool(self.img)
        return obj.shear(angle)
