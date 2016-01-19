import cv2

from color_bins import ColorBins
from utils.utils import Utils
from coordinates import Coordinates


class Image:
    ndarray = None
    coord = None

    def __init__(self, ndarray):
        assert len(ndarray.shape) is 2, "Not a valid image"
        self.ndarray = ndarray
        self.coord = Coordinates(ndarray.shape)

    def __str__(self):
        return str(self.coord)

    def get_original_shape(self):
        return self.ndarray.shape

    def get_custom_shape(self):
        """Ndarray shape as calculated by this object
        Format: (vertical, horizontal)

        >>> import os
        >>> os.path.exists("./img/tests/one_line_lcd.jpg")
        True
        >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg", 0)
        >>> arr is None
        False
        >>> i = Image(arr)
        >>> i.get_original_shape()
        (572, 1310)
        >>> i.get_custom_shape()
        (572, 1310)
        >>> i.get_original_shape() == i.get_custom_shape()
        True
        >>> i.crop(72, 572, 0, 100)
        >>> i.get_custom_shape()
        (500, 100)
        >>> new_arr = arr[72:572, 0:100]
        >>> i.get_custom_shape() == new_arr.shape
        True
        """
        return self.coord.get_shape()

    def get_ndarray(self):
        return self.ndarray[self.coord.left:self.coord.right, self.coord.top:self.coord.bottom]

    def crop(self, coord):
        """Virtual crop.
        API parameters resemble Numpy slicing
        arr[1:30, 3:4]
        arr[start:stop, start:stop]
        crop(1, 30, 3, 4]

        :param coord - instance of Coordinates object

        >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg", 0)
        >>> i = Image(arr)
        >>> i.get_original_shape()
        (572, 1310)
        >>> i.get_custom_shape()  # (vertical, horizontal)
        (572, 1310)
        >>>
        >>> # first crop
        >>> i.crop(72, 300, 210, 1100)
        >>> i.get_custom_shape()
        (228, 890)
        >>> split_one = arr[72:300, 210:1100]
        >>> split_one.shape
        (228, 890)
        >>> split_one.shape == i.get_custom_shape()
        True
        >>> str(i)
        '72 300 210 1100'
        >>>
        >>> # second crop
        >>> i.crop(100, 200, 100, 200)
        >>> i.get_custom_shape()
        (100, 100)
        >>> arr[72:300, 210:1100][100:200, 100:200].shape
        (100, 100)
        >>> arr[72:300, 210:1100][100:200, 100:200].shape == i.get_custom_shape()
        True
        >>> str(i)
        '172 272 310 410'
        """
        self.coord.crop(coord.left, coord.right, coord.top, coord.bottom)

    def crop_borders(self, vertical_percent):
        """
        :param vertical_percent: border width in range from 0 to 1
        :return: void

        >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg", 0)
        >>> i = Image(arr)
        >>> i.get_custom_shape()
        (572, 1310)
        >>> i.crop_borders(0.1)
        >>> i.get_custom_shape()
        (457.59999999999997, 1195.6)
        """
        self.coord.crop_borders(vertical_percent)

    def draw_outline(self, coord):
        """
        :param coord - instance of Coordinates object
        :return ndarray

        >>> # arr = cv2.imread("./img/tests/one_line_lcd.jpg", 0)
        >>> # i = Image(arr)
        >>> # i.draw_outline(100, 300, 100, 300)
        >>> # i.show()
        """
        return cv2.rectangle(self.ndarray,
                             (int(coord.left), int(coord.top)), (int(coord.right), int(coord.bottom)),
                             (0, 255, 0),
                             10)

    def debug(self, window_title):
        modified = self.coord.left is not 0 or self.coord.right is not 0 or self.coord.top is not 0 or self.coord.bottom is not 0
        suffix = "(*)" if modified else ""

        label_partial = "Partial: %s %s" % (self, suffix)
        label_full = "Full: %sx%s" % self.ndarray.shape

        Utils.show_images([self.get_ndarray(), self.ndarray],
                          [label_partial, label_full],
                          window_title)

    def get_color_bins_object(self, count, orientation, interpolation_type=cv2.INTER_AREA):
        return ColorBins(self.get_ndarray(), count, orientation, interpolation_type)
