import numpy as np
import cv2

from utils.utils import Utils


class ColorBins:
    """
    >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg", 0)
    >>> bins = ColorBins(arr, 10, ColorBins.ORIENTATION_HORIZONTAL)
    >>> bins.set_desaturation_params()
    >>> print list(bins.find_ranges(200))
    [(258.07, 920.9300000000001)]
    >>> # bins.show()
    """
    image = None
    bins_count = -1
    orientation = None
    compression = -1

    ORIENTATION_VERTICAL = 0
    ORIENTATION_HORIZONTAL = 1

    color_bins = desaturated = binary_bins = []

    def __init__(self, full_color_ndarray, count, orientation, interpolation_type=cv2.INTER_AREA):
        assert len(full_color_ndarray.shape) == 2, "Not a valid image"
        self.image = full_color_ndarray
        self.bins_count = count
        self.orientation = orientation
        self._set_interpolation_type(interpolation_type)
        measurement = self.image.shape[1] if orientation else self.image.shape[0]
        self.compression = 1.0 * measurement / self.bins_count

    def _set_interpolation_type(self, interpolation_type=cv2.INTER_AREA):
        new_shape = (1, self.bins_count) if self.orientation == self.ORIENTATION_VERTICAL else (self.bins_count, 1)
        vector = cv2.resize(self.image, new_shape, interpolation=interpolation_type)
        self.color_bins = np.transpose(vector) if self.orientation == self.ORIENTATION_VERTICAL else vector

    def set_desaturation_params(self, desaturation_type=cv2.THRESH_BINARY + cv2.THRESH_OTSU, thresh=127, maxval=255):
        assert self.color_bins.shape, "Set interpolation type first"
        th, self.desaturated = cv2.threshold(self.color_bins, thresh, maxval, desaturation_type)

    def _set_thresholding_intensity(self, intensity):
        assert self.desaturated.shape, "Set desaturation params first"
        self.binary_bins = self.desaturated < intensity

    def _split_bins_into_ranges(self):
        assert self.binary_bins.shape, "Set other parameters before calling this method"
        start = False
        finish = False
        is_inside = False
        for index, i in enumerate(self.binary_bins[0]):
            if i and not is_inside:
                is_inside = True
                start = index
            elif not i and is_inside:
                is_inside = False
                finish = index
                yield (start, finish)
        if not finish:
            yield (start, self.bins_count-1)

    def _scale_ranges(self, ranges, overflow_size):
        for left, right in ranges:
            # print 'scale', region, bins_count, offset
            left_scaled = max(0, (left - overflow_size) * self.compression)
            right_scaled = min(self.bins_count * self.compression, right + overflow_size) * self.compression
            # print 'scaled:', left, right
            yield left_scaled, right_scaled

    def find_ranges(self, thresholding_intensity=127, overflow_percent=0.03):
        """
        :param thresholding_intensity:  0 to 255
        :param overflow_percent:    0 to 1
        :return:     [(10, 100), (150, 300)]
        """
        self._set_thresholding_intensity(thresholding_intensity)
        r = self._split_bins_into_ranges()
        return self._scale_ranges(r, overflow_percent)

    def debug(self, window_title):
        # colored, monochrome and vector
        assert len(self.color_bins), "Run other methods first"
        linspace = np.linspace(0, 255, 255)
        reference_colors = [linspace, linspace, linspace]
        Utils.show_images([self.image, self.color_bins, self.binary_bins, reference_colors],
                          ["Full color", "Color bins", "Binary bins", "Color range 0..255"],
                          window_title)
