import cv2

from utilities.projection import Projection
from utilities.rectangle import Rectangle
from utilities.visualizer import Visualizer


class AreaFactory:
    """
    >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> f = AreaFactory(arr, Projection.TYPE_VERTICAL, Rectangle.from_ndarray(arr), 100)
    >>> list(f.find_areas())
    [(0:572, 0:1310)]
    >>> f.debug('area factory vertical')
    >>>
    >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> f = AreaFactory(arr, Projection.TYPE_HORIZONTAL, Rectangle.from_ndarray(arr), 100)
    >>> list(f.find_areas())
    [(0:572, 169:1009)]
    >>> f.debug('area factory horizontal')
    """
    img = None
    projection = None
    basis = None
    measurement = -1

    def __init__(self, full_color_image, orientation, basis, bins_count):
        Projection(full_color_image, orientation, 0.2).debug()
        self.img = full_color_image
        self.projection = Projection(full_color_image, orientation, 0.2)
        self.basis = basis or Rectangle.from_ndarray(full_color_image)
        self.measurement = full_color_image.shape[orientation]

    def find_areas(self, overflow_percent=0.03):
        """
        :param overflow_percent:    0 to 1
        :return:     [(10, 100), (150, 300)]
        """
        r = self._split_projection_into_ranges()
        s = self._scale_ranges(r, overflow_percent)
        return self._convert_ranges_into_coordinates(self.basis, s)

    def _split_projection_into_ranges(self):
        peaks = list(self.projection.get_peaks())
        if len(peaks) % 2:
            peaks = peaks[1:]
        if len(peaks) < 2:
            yield 0, self.projection.measurement
        else:
            for i in xrange(0, len(peaks), 2):
                yield peaks[i][0], peaks[i+1][0]

    def _scale_ranges(self, ranges, overflow_size):
        compression = 1.0 * self.measurement / self.projection.measurement
        for first, second in ranges:
            first_scaled = max(0, (first - overflow_size) * compression)
            second_scaled = min(self.projection.measurement, second + overflow_size) * compression
            yield first_scaled, second_scaled

    def _convert_ranges_into_coordinates(self, basis, ranges):
        for first_scaled, second_scaled in ranges:
            if self.projection.orientation == Projection.TYPE_VERTICAL:
                yield Rectangle(basis.top + first_scaled, basis.top + second_scaled,
                                basis.left, basis.left + basis.width)
            else:
                yield Rectangle(basis.top, basis.top + basis.height,
                                basis.left + first_scaled, basis.left + second_scaled)

    def debug(self, window_title="Area Factory"):
        Visualizer.outline(self.img, self.find_areas(), window_title=window_title)
