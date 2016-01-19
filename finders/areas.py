import cv2

from image.color_bins import ColorBins
from image.image import Image


class DigitsAreaFinder:
    """
    >>> import os
    >>> row_path = './img/tests/row.jpg'
    >>> os.path.isfile(row_path)
    True
    >>> img = cv2.imread(row_path, 0)
    >>> img.shape
    (495, 1257)
    >>> i = Image(img)
    >>> r = DigitsAreaFinder(i)
    >>> e = list(r.extract())
    >>> e
    [(121.929, 1009.371)]
    >>> r.debug("before loop")
    >>> for area in e:
    ...     print area
    ...     # todo save state
    ...     i.crop(0, 0, area[0], area[1])
    ...     i.draw_outline()
    ...     i.reset()
    ...     # print "inside loop", r.debug("inside loop")
    >>> r.debug("after loop")
    """
    img = bins = None
    bins_count = 10
    threshold = 200

    def __init__(self, image_object):
        self.img = image_object

    def extract(self):
        self.bins = self.img.get_color_bins_object(10, ColorBins.ORIENTATION_HORIZONTAL)
        self.bins.set_desaturation_params()
        ranges = self.bins.find_ranges(self.threshold)
        for reg in ranges:
            # result = self.black_and_white[:, reg[0]:reg[1]]
            # yield reg[0], reg[1], self.top, self.bottom
            yield reg

    def debug(self, window_title):
        print '\n=== DigitsAreaFinder'
        print 'custom shape', self.img.get_custom_shape()
        print 'numpy shape', self.img.get_original_shape()
        print 'internals', self.img
        self.img.debug(window_title)
        if self.bins:
            print 'binary bins', self.bins.binary_bins
            self.bins.debug(window_title)
        print '\n'


class RowAreasFinder:
    # """
    # >>> import os
    # >>> path = "./img/tests/one_line_lcd.jpg"
    # >>> os.path.isfile(path)
    # True
    # >>> img = cv2.imread(path, 0)
    # >>> img.shape
    # (572, 1310)
    # >>> i = Image(img)
    # >>> r = RowAreasFinder(i)
    # >>> e = r.extract()
    # >>> list(e)
    # [(0, 0, 14.3, 529.1)]
    # """
    img = bins = None
    bins_count = 20
    threshold = 200

    def __init__(self, image_object):
        self.img = image_object

    def extract(self):
        self.bins = self.img.get_color_bins_object(20, ColorBins.ORIENTATION_VERTICAL)
        self.bins.set_desaturation_params()
        ranges = self.bins.find_ranges(self.threshold)
        for r in ranges:
            # todo crop
            f = DigitsAreaFinder(self)
            for t in f.extract():
                yield t

    def debug(self, window_title):
        print '\n=== RowAreasFinder'
        print 'custom shape', self.img.get_custom_shape()
        print 'numpy shape', self.img.get_numpy_shape()
        print 'internals', self.img
        print 'binary bins', self.bins.binary_bins
        self.img.debug(window_title)
        self.bins.debug(window_title)
