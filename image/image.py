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
        self.coord = Coordinates.from_ndarray(self.ndarray)

    def __str__(self):
        return str(self.coord)

    def reset(self):
        self.coord = Coordinates.from_ndarray(self.ndarray)

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
        >>> i.crop(Coordinates(72, 572, 0, 1000))
        >>> i.get_custom_shape()
        (500, 1000)
        >>> new_arr = arr[72:572, 0:1000]
        >>> i.get_custom_shape() == new_arr.shape
        True
        """
        return self.coord.get_shape()

    def get_ndarray(self):
        return self.ndarray[self.coord.top:self.coord.bottom, self.coord.left:self.coord.right]

    def crop(self, coord):
        """Virtual crop

        :param coord - instance of Coordinates object
        """
        self.coord.crop(coord)

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

    def debug(self, window_title="Image"):
        modified = self.coord.left is not 0 or self.coord.right is not 0 or self.coord.top is not 0 or self.coord.bottom is not 0
        suffix = "(*)" if modified else ""

        label_partial = "Partial: %s %s" % (self, suffix)
        label_full = "Full: %sx%s" % self.ndarray.shape

        Utils.show_images([self.get_ndarray(), self.ndarray],
                          [label_partial, label_full],
                          window_title)

    def get_color_bins_object(self, count, orientation, interpolation_type=cv2.INTER_AREA):
        return ColorBins(self.get_ndarray(), count, orientation, interpolation_type)

# todo travis
