import cv2
import numpy as np
from visible.projection import Projection
from italics.shear_tool import ShearTool
from utils.utils import Utils


class FontNormalizer:
    """
    >>> before = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> obj = FontNormalizer(before)
    >>> incl = obj.find_inclination()
    >>> incl  # todo it should be closer to 8.0
    3.0
    >>> first = obj.normalize_font(incl)
    >>> Utils.show_images([before, first], ["Before", "First"], "Font Normalizer")
    >>> obj2 = FontNormalizer(first)
    >>> obj2.find_inclination()
    0.0
    """
    img = None

    def __init__(self, img):
        assert isinstance(img, np.ndarray)
        self.img = img

    def calculate_weights(self):
        # todo refactor into logn
        for angle in np.arange(-30., 30., 1.):
            yield angle, self.calculate_single_weight(angle)

    def calculate_single_weight(self, angle):
        """Finds the most strong gradient in the image
        """
        obj = ShearTool(self.img)
        corrected = obj.shear(angle)
        p = Projection(corrected, self.img.shape[1], Projection.TYPE_HORIZONTAL)
        projection = p.make_binary_projection()
        # p.debug()
        return max(np.gradient(projection))

    def find_inclination(self):
        weights = self.calculate_weights()
        weights = list(weights)
        values = [val for (idx, val) in weights]
        Utils.plot_projection(values, self.img)
        val, idx = max((abs(val), idx) for (idx, val) in weights)
        # print val, idx
        return idx

    def normalize_font(self, angle):
        obj = ShearTool(self.img)
        return obj.shear(angle)
