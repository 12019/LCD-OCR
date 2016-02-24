import numpy as np

import cv2

from coordinates import Coordinates
from croppable_image import CroppableImage
from visible.projection import Projection
from visible.outline import Outline
from visible.single_row import SingleRow


class LCD:
    """
    >>> img = cv2.imread("assets/img/doctests/single_line_lcd.jpg", 0)
    >>> r = LCD(img)
    >>> areas = list(r.extract_rows())
    >>> areas
    [(57:514, 173:1016)]
    >>> r.debug()
    """
    img = factory = None
    bins_count = 20
    threshold = 200

    def __init__(self, ndarray, basis=None):
        assert isinstance(ndarray, np.ndarray)
        assert (basis is None) or isinstance(basis, Coordinates)
        ndarray = cv2.equalizeHist(ndarray)
        self.img = CroppableImage(ndarray, basis)
        self.img.crop_borders(0.1)

    def extract_rows(self):
        self.factory = self.img.get_area_factory(self.bins_count, Projection.TYPE_VERTICAL)
        return self.factory.find_areas()

    def extract_areas(self):
        for row in self.extract_rows():
            area = SingleRow(self.img.ndarray, row)
            for coord in area.extract_digits_areas():
                yield coord

    def debug(self, window_title="Find rows"):
        Outline(
            self.img.ndarray
            , self.extract_rows()
            # , self.extract_areas()
        ).show(window_title)
