import numpy as np
import cv2

from utils.utils import Utils


class ColorBins:
    """
    >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg", 0)
    >>> bins = ColorBins(arr)
    >>> bins.apply_threshold()
    >>> bins.get_bins(20, True)
    >>> bins.show()
    array([234, 240, 252,  98, 145,  69, 208, 115, 120, 111, 126, 191,  74,
           144, 170, 254, 254, 252, 255, 255], dtype=uint8)
    """
    full_color = monochrome = None
    vector = []

    def __init__(self, full_color_ndarray):
        self.full_color = full_color_ndarray

    def apply_threshold(self, threshold_type=cv2.THRESH_OTSU, thresh=127, maxval=255):
        th, self.monochrome = cv2.threshold(self.full_color, thresh, maxval, threshold_type)
        # print self.monochrome

    def get_bins(self, count, is_horizontal, interpolation_type=cv2.INTER_AREA):
        assert self.monochrome.shape, "Apply threshold first"
        new_shape = (1, count) if not is_horizontal else (count, 1)
        vector = cv2.resize(self.monochrome, new_shape, interpolation=interpolation_type)
        vector = np.transpose(vector) if not is_horizontal else vector
        self.vector = vector
        return self.vector[0]

    def show(self):
        # colored, monochrome and vector
        assert len(self.vector), "Run other methods first"
        Utils.show_images([self.full_color, self.monochrome, self.vector],
                          ["Full color", "Monochrome", "Vector"])