import numpy as np

import cv2

from croppable.area_factory import AreaFactory
from utilities.rectangle import Rectangle
from utilities.plotter import Plotter


class CroppableImage:
    ndarray = None
    coord = None
    latest_projection = None  # may be None

    def __init__(self, ndarray, basis=None):
        assert isinstance(ndarray, np.ndarray)
        assert len(ndarray.shape) is 2, "Not a valid visible"
        self.ndarray = ndarray
        if basis:
            self.coord = basis
        else:
            self.coord = Rectangle.from_ndarray(self.ndarray)

    def reset(self):
        self.coord = Rectangle.from_ndarray(self.ndarray)

    def get_original_shape(self):
        return self.ndarray.shape

    def get_custom_shape(self):
        """Ndarray shape as calculated by this object
        Format: (vertical, horizontal)

        >>> import os
        >>> os.path.exists('assets/img/doctests/single_line_lcd.jpg')
        True
        >>> arr = cv2.imread('assets/img/doctests/single_line_lcd.jpg', 0)
        >>> arr is None
        False
        >>> i = CroppableImage(arr)
        >>> i.get_original_shape()
        (572, 1310)
        >>> i.get_custom_shape()
        (572, 1310)
        >>> i.get_original_shape() == i.get_custom_shape()
        True
        >>> i.crop(Rectangle(72, 572, 0, 1000))
        >>> i.get_custom_shape()
        (500, 1000)
        >>> new_arr = arr[72:572, 0:1000]
        >>> i.get_custom_shape() == new_arr.shape
        True
        >>> # i.debug('custom shape test')
        """
        return self.coord.get_shape()

    def get_ndarray(self):
        return self.ndarray[
               self.coord.top:self.coord.top+self.coord.height,
               self.coord.left:self.coord.left+self.coord.width]

    def crop(self, coord):
        """Virtual crop

        :param coord - instance of Coordinates object
        """
        self.coord.crop(coord)

    def get_subimages(self, orientation):
        """
        >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
        >>> ci = CroppableImage(arr)
        >>> from projection.abstract import Projection
        >>> subimages = list(ci.get_subimages(Projection.TYPE_HORIZONTAL))
        >>> subimages
        [<CroppableImage 0+572, 184+838 out of (572, 1310)>]
        >>> # subimages[0].debug("subimage 1")
        """
        factory = AreaFactory(self.ndarray, orientation, self.coord)
        self.latest_projection = factory.get_projection()
        areas = factory.find_areas()
        return [CroppableImage(self.ndarray, area) for area in areas]

    # def crop_borders(self, vertical_percent):
    #     """
    #     :param vertical_percent: border width in range from 0 to 1
    #     :return: void
    #
    #     >>> arr = cv2.imread("assets/img/doctests/single_line_lcd.jpg", 0)
    #     >>> i = CroppableImage(arr)
    #     >>> i.get_custom_shape()
    #     (572, 1310)
    #     >>> i.crop_borders(0.1)
    #     >>> i.get_custom_shape()
    #     (457.59999999999997, 1195.6)
    #     """
    #     self.coord.crop_borders(vertical_percent)

    def __repr__(self):
        return "<CroppableImage %s+%s, %s+%s out of %s>" % \
               (self.coord.top, self.coord.height, self.coord.left, self.coord.width,
                self.ndarray.shape)

    def __str__(self):
        return str(self.coord)

    def debug(self, window_title="Image"):
        Plotter.outline_rectangles(self.ndarray, [self.coord], [], window_title, self.latest_projection)
