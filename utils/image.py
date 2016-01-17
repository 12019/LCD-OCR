import cv2
from utils.utils import Utils
from utils.color_bins import ColorBins


class Image:
    ndarray = None
    left = 0
    right = 0
    top = 0
    bottom = 0

    def __init__(self, ndarray):
        assert ndarray.shape, "Not a valid image"
        self.ndarray = ndarray
        self.left = self.top = 0
        self.right, self.bottom, d = self.ndarray.shape

    def __str__(self):
        return "%s %s %s %s" % (self.left, self.right, self.top, self.bottom)

    def get_original_shape(self):
        return self.ndarray.shape

    def get_custom_shape(self):
        """Ndarray shape as calculated by this object

        >>> import os
        >>> os.path.exists("./img/tests/one_line_lcd.jpg")
        True
        >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg")
        >>> arr is None
        False
        >>> i = Image(arr)
        >>> i.get_original_shape()
        (572, 1310, 3)
        >>> i.get_custom_shape()
        (572, 1310, 3)
        >>> i.get_original_shape() == i.get_custom_shape()
        True
        """
        return self.right - self.left, self.bottom - self.top, 3

    def get_ndarray(self):
        return self.ndarray[self.left:self.right, self.top:self.bottom]

    def revert(self):
        self.left = self.right = self.top = self.bottom = 0

    def crop(self, left, right, top, bottom):
        """Virtual crop.
        API parameters resemble Numpy slicing
        arr[1:30, 3:4]
        arr[start:stop, start:stop]
        crop(1, 30, 3, 4]

        >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg")
        >>> i = Image(arr)
        >>> i.get_custom_shape()
        (572, 1310, 3)
        >>> i.crop(72, 572, 0, 1000)
        >>> i.get_custom_shape()
        (500, 1000, 3)
        >>> str(i)
        '72 572 0 1000'
        >>> i.crop(100, 200, 100, 200)
        >>> i.get_custom_shape()
        (100, 100, 3)
        >>> str(i)
        '172 272 100 200'
        """
        self.left += left
        self.right += left + right
        self.top += top
        self.bottom += bottom + top

    def crop_borders(self, percent):
        """
        >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg")
        >>> i = Image(arr)
        >>> i.get_custom_shape()
        ()
        >>> i.crop_borders()
        >>> i.get_custom_shape()
        ()
        :param percent: border width in range from 0 to 1
        :return: void
        """
        pixels = self.get_custom_shape()[0] * percent
        self.left += pixels
        self.right -= pixels
        self.top += pixels
        self.bottom -= pixels

    def draw_outline(self, left, right, top, bottom):
        # """
        # >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg")
        # >>> i = Image(arr)
        # >>> i.draw_outline(100, 300, 100, 300)
        # >>> i.show()
        # """
        cv2.rectangle(self.get_ndarray(), (left, top), (right, bottom), (0, 255, 0), 10)

    def show(self):
        Utils.show_image(self.get_ndarray())

    def get_color_bins_object(self):
        return ColorBins(self.get_ndarray())
