import numpy as np

import cv2

from coordinates import Coordinates
from croppable_image import CroppableImage
from visible.outline import Outline
from visible.projection import Projection
from visible.single_row import SingleRow


class LCD:
    """
    >>> import os
    >>> path = "assets/img/doctests/single_line_lcd.jpg"
    >>> os.path.isfile(path)
    True
    >>> img = cv2.imread(path, 0)
    >>> img.shape
    (572, 1310)
    >>> r = LCD(img)
    >>> areas = list(r.extract_digits_area())
    >>> areas
    [(57:514, 173:1016)]
    >>> o = Outline(img, areas)
    >>> # o.show()
    """
    img = projection = None
    bins_count = 20
    threshold = 200

    def __init__(self, ndarray, basis=None):
        assert isinstance(ndarray, np.ndarray)
        assert (basis is None) or isinstance(basis, Coordinates)
        ndarray = cv2.equalizeHist(ndarray)
        self.img = CroppableImage(ndarray, basis)
        self.img.crop_borders(0.1)

    def extract_digits_area(self):
        self.projection = self.img.get_projections_object(self.bins_count, Projection.TYPE_VERTICAL)
        self.projection.make_binary_projection(cv2.INTER_CUBIC, cv2.THRESH_BINARY, self.threshold)
        areas = self.projection.find_areas(self.img.coord)
        # self.debug()
        for r in areas:
            f = SingleRow(self.img.ndarray, r)
            for t in f.extract_digits_areas():
                yield t

    def debug(self, window_title="Find rows"):
        print '\n=== LCD class'
        print 'custom shape', self.img.get_custom_shape()
        print 'numpy shape', self.img.get_original_shape()
        print 'internals', self.img
        print 'color bins', self.projection.colorful_projection
        print 'binary bins', self.projection.binary_projection
        self.img.debug(window_title)
        self.projection.debug(window_title)


