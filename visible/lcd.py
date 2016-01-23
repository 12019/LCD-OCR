import numpy as np

from coordinates import Coordinates
from croppable_image import CroppableImage
from visible.projection import Projection
from visible.single_row import SingleRow
import cv2


class LCD:
    """
    >>> import os
    >>> path = "./img/tests/one_line_lcd.jpg"
    >>> os.path.isfile(path)
    True
    >>> img = cv2.imread(path, 0)
    >>> img.shape
    (572, 1310)
    >>> r = LCD(img)
    >>> areas = list(r.extract_digits_area())
    >>> areas
    [(27:515, 258:920)]
    >>> # o = Outline(img, areas)
    >>> # o.show()
    """
    img = bins = None
    bins_count = 20
    threshold = 200

    def __init__(self, ndarray, basis=None):
        assert isinstance(ndarray, np.ndarray)
        assert (basis is None) or isinstance(basis, Coordinates)
        self.img = CroppableImage(ndarray, basis)
        # self.img.crop_borders(0.1)

    def extract_digits_area(self):
        self.bins = self.img.get_color_bins_object(self.bins_count, Projection.PROJECTION_VERTICAL)
        self.bins.make_binary_projection(self.threshold)
        areas = self.bins.find_areas(self.img.coord)
        # self.debug()
        for r in areas:
            f = SingleRow(self.img.ndarray, r)
            for t in f.extract_digits_areas():
                yield t

    def debug(self, window_title="Find rows"):
        print '\n=== RowAreasFinder'
        print 'custom shape', self.img.get_custom_shape()
        print 'numpy shape', self.img.get_original_shape()
        print 'internals', self.img
        print 'color bins', self.bins.color_bins
        print 'binary bins', self.bins.binary_bins
        self.img.debug(window_title)
        self.bins.debug(window_title)


