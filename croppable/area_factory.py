import cv2

from projection.abstract import Projection
from projection.binary import BinaryProjection
from utilities.plotter import Plotter
from utilities.rectangle import Rectangle


class AreaFactory:
    """
    >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> f = AreaFactory(arr, Projection.TYPE_VERTICAL, Rectangle.from_ndarray(arr))
    >>> list(f.find_areas())
    [<Rectangle 0+572, 0+1310>]
    >>> # f.debug('area factory vertical')
    >>>
    >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> f = AreaFactory(arr, Projection.TYPE_HORIZONTAL, Rectangle.from_ndarray(arr))
    >>> list(f.find_areas())
    [<Rectangle 0+572, 184+838>]
    >>> # f.debug('area factory horizontal')
    """
    img = None
    projection = None
    basis = None
    measurement = -1

    def __init__(self, full_color_image, orientation, basis):
        # Projection(full_color_image, orientation, 0.2).debug()
        self.img = full_color_image
        self.projection = BinaryProjection(full_color_image, orientation)
        self.basis = basis or Rectangle.from_ndarray(full_color_image)
        self.measurement = full_color_image.shape[orientation]

    def find_areas(self, overflow_percent=0.03):
        """
        :param overflow_percent:    0 to 1
        :return:     [(10, 100), (150, 300)]
        """
        ranges = self._split_peaks_into_ranges()
        return self._convert_ranges_into_coordinates(self.basis, ranges)

    def get_projection(self):
        return self.projection.get_projection()

    def _split_peaks_into_ranges(self):
        peaks = list(self.projection.get_peaks())
        if len(peaks) % 2:
            peaks = peaks[1:]
        if len(peaks) < 2:
            yield 0, self.projection.measurement
        else:
            for i in xrange(0, len(peaks), 2):
                yield peaks[i][0], peaks[i+1][0]

    def _convert_ranges_into_coordinates(self, basis, ranges):
        for first_scaled, second_scaled in ranges:
            if self.projection.orientation == Projection.TYPE_VERTICAL:
                yield Rectangle(basis.top + first_scaled, second_scaled - first_scaled,
                                basis.left, basis.width)
            else:
                yield Rectangle(basis.top, basis.height,
                                basis.left + first_scaled, second_scaled - first_scaled)

    def debug(self, window_title="Area Factory"):
        areas = list(self.find_areas())  # todo why casting to list is required?
        Plotter.outline_rectangles(self.img, areas, window_title=window_title)
