import cv2
import numpy as np
from visible.projection import Projection
from italics.shear_tool import ShearTool
from utils.utils import Utils


class FontNormalizer:
    """
    >>> before = cv2.imread('img/tests/one_line_lcd.jpg', 0)
    >>> obj = FontNormalizer(before)
    >>> obj.find_inclination()
    8.0
    >>> first = obj.normalize_font(8.0)
    >>> Utils.show_images([before, first], ["Before", "First"], "Font Normalizer")
    >>> obj2 = FontNormalizer(first)
    >>> obj2.find_inclination()
    """
    img = None

    def __init__(self, img):
        assert isinstance(img, np.ndarray)
        self.img = img

    def calculate_weights(self):
        for i in np.arange(-30., 30., 5.):
            obj = ShearTool(self.img)
            corrected = obj.shear(i)
            projection = Projection(corrected, self.img.shape[1], Projection.TYPE_HORIZONTAL)
            projection.make_colorful_projection(cv2.INTER_AREA)
            # Utils.plot_projection(projection.colorful_projection, projection.image)
            yield i, np.sum(projection.colorful_projection)

    def find_inclination(self):
        weights = self.calculate_weights()
        weights = list(weights)
        # print weights
        val, idx = max((val, idx) for (idx, val) in weights)
        return idx

    def normalize_font(self, angle):
        obj = ShearTool(self.img)
        return obj.shear(angle)
