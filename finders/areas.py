import cv2

from image.color_bins import ColorBins
from image.image import Image
from image.coordinates import Coordinates
from image.outline import Outline
import numpy as np


class DigitsAreaFinder:
    # """
    # >>> import os
    # >>> row_path = './img/tests/row.jpg'
    # >>> os.path.isfile(row_path)
    # True
    # >>> img = cv2.imread(row_path, 0)
    # >>> img.shape
    # (495, 1257)
    # >>> r = DigitsAreaFinder(img)
    # >>> e = list(r.extract())
    # >>> e
    # [(0:495, 121:1009)]
    # >>> # r.debug()
    # >>> o = Outline(img, e)
    # >>> o.show()
    # """
    img = bins = None
    bins_count = 20
    threshold = 200

    def __init__(self, ndarray, basis=None):
        assert isinstance(ndarray, np.ndarray)
        assert (basis is None) or isinstance(basis, Coordinates)
        self.img = Image(ndarray, basis)

    def extract(self):
        self.bins = self.img.get_color_bins_object(self.bins_count, ColorBins.ORIENTATION_HORIZONTAL)
        self.bins.threshold(self.threshold)
        area = self.bins.find_areas(self.img.coord)
        self.debug()
        for area in area:
            yield area

    def debug(self, window_title="Digits Area Finder"):
        print '\n=== DigitsAreaFinder'
        print 'custom shape', self.img.get_custom_shape()
        print 'numpy shape', self.img.get_original_shape()
        print 'internals', self.img
        self.img.debug(window_title)
        if self.bins:
            # print 'color bins', self. bins.color_bins
            print 'binary bins', self.bins.binary_bins
            self.bins.debug(window_title)
        print '\n'


class RowAreasFinder:
    """
    >>> import os
    >>> path = "./img/tests/one_line_lcd.jpg"
    >>> os.path.isfile(path)
    True
    >>> img = cv2.imread(path, 0)
    >>> img.shape
    (572, 1310)
    >>> r = RowAreasFinder(img)
    >>> areas = list(r.extract())
    >>> areas
    [(0, 0, 14.3, 529.1)]
    >>> o = Outline(img, areas)
    >>> o.show()
    """
    img = bins = None
    bins_count = 20
    threshold = 200

    def __init__(self, ndarray, basis=None):
        assert isinstance(ndarray, np.ndarray)
        assert (basis is None) or isinstance(basis, Coordinates)
        self.img = Image(ndarray, basis)
        # self.img.crop_borders(0.1)

    def extract(self):
        self.bins = self.img.get_color_bins_object(self.bins_count, ColorBins.ORIENTATION_VERTICAL)
        self.bins.threshold(self.threshold)
        areas = self.bins.find_areas(self.img.coord)
        # self.debug()
        for r in areas:
            print r
            f = DigitsAreaFinder(self.img.ndarray, r)
            for t in f.extract():
                yield t

    def debug(self, window_title="RowAreasFinder"):
        print '\n=== RowAreasFinder'
        print 'custom shape', self.img.get_custom_shape()
        print 'numpy shape', self.img.get_original_shape()
        print 'internals', self.img
        print 'color bins', self.bins.color_bins
        print 'binary bins', self.bins.binary_bins
        self.img.debug(window_title)
        self.bins.debug(window_title)
