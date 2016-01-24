import numpy as np

from projection import Projection
from coordinates import Coordinates
from croppable_image import CroppableImage
import cv2
from visible.outline import Outline


class SingleRow:
    """
    >>> import os
    >>> row_path = './img/tests/row.jpg'
    >>> os.path.isfile(row_path)
    True
    >>> img = cv2.imread(row_path, 0)
    >>> img.shape
    (495, 1257)
    >>> r = SingleRow(img)
    >>> e = list(r.extract_digits_areas())
    >>> e
    [(0:495, 121:1009)]
    >>> # r.debug()
    >>> # o = Outline(img, e)
    >>> # o.show()
    """
    img = bins = None
    bins_count = 10
    threshold = 200
    areas = None

    def __init__(self, ndarray, basis=None):
        assert isinstance(ndarray, np.ndarray)
        assert (basis is None) or isinstance(basis, Coordinates)
        self.img = CroppableImage(ndarray, basis)

    def extract_digits_areas(self):
        self.bins = self.img.get_projections_object(self.bins_count, Projection.TYPE_HORIZONTAL)
        self.bins.make_binary_projection(self.threshold)
        self.areas = self.bins.find_areas(self.img.coord)
        return self.areas

    def debug(self, window_title="Find digits"):
        print '\n=== Single Row'
        print 'custom shape', self.img.get_custom_shape()
        print 'numpy shape', self.img.get_original_shape()
        print 'internals', self.img
        # self.img.debug(window_title)
        if self.bins:
            # print 'color bins', self. bins.color_bins
            print 'binary bins', self.bins.binary_bins
            self.bins.debug(window_title)
        print '\n'

