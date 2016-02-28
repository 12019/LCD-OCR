import cv2

from croppable.croppable_image import CroppableImage
from projection.abstract import Projection
from utilities.plotter import Plotter
from visible.single_row import SingleRow


class LCD:
    """
    >>> img = cv2.imread("assets/img/doctests/single_line_lcd.jpg", 0)
    >>> obj = LCD(CroppableImage(img))
    >>> areas = list(obj.extract_rows())
    >>> areas
    [<CroppableImage 0+572, 0+1310 out of (572, 1310)>]
    >>> digits = list(obj.extract_areas_with_digits())
    >>> digits
    [<CroppableImage 0+572, 184+838 out of (572, 1310)>]
    >>> # obj.debug()
    >>>
    >>> img = cv2.imread("assets/img/doctests/multi_line_lcd.jpg", 0)  # todo multiline display thresholding
    >>> obj = LCD(CroppableImage(img))
    >>> digits = list(obj.extract_areas_with_digits())
    >>> digits
    [<CroppableImage 0+344, 0+384 out of (344, 384)>]
    >>> # obj.debug()  # todo
    """
    image = None

    def __init__(self, image):
        assert isinstance(image, CroppableImage)
        self.image = image

    def extract_rows(self):
        return self.image.get_subimages(Projection.TYPE_VERTICAL)

    def extract_areas_with_digits(self):
        rows = [SingleRow(image) for image in self.extract_rows()]
        return sum([row.extract_digits_areas() for row in rows], [])

    def debug(self, window_title="Find rows"):
        Plotter.outline_subimages(self.image, self.extract_rows(), window_title=window_title)

    def __repr__(self):
        return "<LCD %sx%s>" % self.image.ndarray.shape
