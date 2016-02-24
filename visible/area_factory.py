import cv2
from visible.projection import Projection
from visible.outline import Outline
from coordinates import Coordinates


class AreaFactory:
    """
    >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> f = AreaFactory(arr, 100, Coordinates.from_ndarray(arr), Projection.TYPE_VERTICAL)
    >>> list(f.find_areas())
    [(0:572, 0:1310)]
    >>> f.debug()
    >>>
    >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
    >>> f = AreaFactory(arr, 100, Coordinates.from_ndarray(arr), Projection.TYPE_HORIZONTAL)
    >>> list(f.find_areas())
    [(0:572, 169:1009)]
    >>> f.debug()
    >>> # todo more than one area
    """
    img = None
    projection = None
    basis = None
    measurement = -1

    def __init__(self, full_color_image, bins_count, basis, orientation):  # todo change signature
        # Projection(full_color_image, bins_count, orientation, 0.2).debug()
        self.img = full_color_image
        self.projection = Projection(full_color_image, bins_count, orientation, 0.2)
        self.basis = basis or Coordinates.from_ndarray(full_color_image)
        self.measurement = full_color_image.shape[orientation]

    def find_areas(self, overflow_percent=0.03):  # todo gradients
        """
        :param overflow_percent:    0 to 1
        :return:     [(10, 100), (150, 300)]
        """
        r = self._split_projection_into_ranges_2()
        s = self._scale_ranges(r, overflow_percent)
        return self._convert_ranges_into_coordinates(self.basis, s)

    def _split_projection_into_ranges_2(self):
        peaks = list(self.projection.get_peaks())
        if len(peaks) % 2:
            peaks = peaks[1:]
        if len(peaks) < 2:
            yield 0, self.projection.bins_count
        else:
            for i in xrange(0, len(peaks), 2):
                yield peaks[i][0], peaks[i+1][0]

    def _split_projection_into_ranges(self):
        """Analyzing gradient(vector) / vector
        Assuming there is an odd number of dominant peaks.
        3, 5 or 7 in total
        :return:
        """
        # indices, L_sorted = zip(*sorted(enumerate(self.peaks_function), key=itemgetter(1), reverse=True))
        # print indices[:6], L_sorted[:6]
        # start = False
        # stop = False
        # is_inside = False
        # for index, i in enumerate(np.round(self.histogram)):
        #     if i and not is_inside:
        #         is_inside = True
        #         start = index
        #     elif not i and is_inside:
        #         is_inside = False
        #         stop = index
        #         yield (start, stop)
        # if not stop:
        #     yield (start, self.bins_count)

    def _scale_ranges(self, ranges, overflow_size):
        compression = 1.0 * self.measurement / self.projection.bins_count
        for left, right in ranges:
            # print 'scale', region, bins_count, offset
            left_scaled = max(0, (left - overflow_size) * compression)
            right_scaled = min(self.projection.bins_count, right + overflow_size) * compression
            # print 'scaled:', left, right
            yield left_scaled, right_scaled

    def _convert_ranges_into_coordinates(self, basis, ranges):
        for left_scaled, right_scaled in ranges:
            if self.projection.orientation == Projection.TYPE_VERTICAL:
                yield Coordinates(basis.top + left_scaled, basis.top + right_scaled, basis.left, basis.right)
            else:
                yield Coordinates(basis.top, basis.bottom, basis.left + left_scaled, basis.left + right_scaled)

    def debug(self):
        Outline(self.img, self.find_areas()).show()
