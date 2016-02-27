import cv2

from croppable.croppable_image import CroppableImage
from croppable.projection import Projection
from utilities.plotter import Plotter


class SingleRow:
    """
    >>> img = cv2.imread('assets/img/doctests/row.jpg', 0)
    >>> r = SingleRow(CroppableImage(img))
    >>> e = list(r.extract_digits_areas())
    >>> e
    [(0:495, 169:850) out of (495, 1257)]
    >>> # r.debug()  # todo why offset
    """
    image = None

    def __init__(self, image):
        assert isinstance(image, CroppableImage)
        self.image = image

    def extract_digits_areas(self):
        return self.image.get_subimages(Projection.TYPE_HORIZONTAL)

    def debug(self, window_title="Find digits"):
        Plotter.outline_subimages(self.image, self.extract_digits_areas(), window_title=window_title)
