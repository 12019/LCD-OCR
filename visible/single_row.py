import numpy as np

import cv2

from coordinates import Coordinates
from visible.area_factory import AreaFactory
from visible.outline import Outline
from visible.projection import Projection


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
        assert (basis is None) or isinstance(basis, Coordinates)
        self.img = ndarray
        self.factory = AreaFactory(ndarray, 100, basis, Projection.TYPE_HORIZONTAL)

    def extract_digits_areas(self):
        return self.factory.find_areas()

    def debug(self, window_title="Find digits"):
        Outline(self.img, self.extract_digits_areas()).show(window_title)
