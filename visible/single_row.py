import numpy as np

import cv2

from utilities.area_factory import AreaFactory
from utilities.projection import Projection
from utilities.rectangle import Rectangle
from utilities.visualizer import Visualizer


class SingleRow:
    """
    >>> import os
    >>> row_path = 'assets/img/doctests/row.jpg'
    >>> os.path.isfile(row_path)
    True
    >>> img = cv2.imread(row_path, 0)
    >>> img.shape
    (495, 1257)
    >>> r = SingleRow(img)
    >>> e = list(r.extract_digits_areas())
    >>> e
    [(0:495, 150:1005)]
    >>> r.debug()
    """
    img = bins = None
    bins_count = 10
    threshold = 200
    areas = None
    factory = None

    def __init__(self, ndarray, basis=None):
        assert isinstance(ndarray, np.ndarray)
        assert (basis is None) or isinstance(basis, Rectangle)
        self.img = ndarray
        self.factory = AreaFactory(ndarray, Projection.TYPE_HORIZONTAL, basis, 100)

    def extract_digits_areas(self):
        return self.factory.find_areas()

    def debug(self, window_title="Find digits"):
        Visualizer.outline(self.img, self.extract_digits_areas(), window_title=window_title)
