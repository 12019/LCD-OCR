import cv2

from projection.abstract import Projection


class ColorfulProjection(Projection):

    def get_projection(self, interpolation_type=cv2.INTER_CUBIC):
        """List of average pixel values along the axis

        >>> arr = cv2.imread('assets/img/doctests/digits.jpg', 0)
        >>> # horizontal
        >>> bins = ColorfulProjection(arr, Projection.TYPE_HORIZONTAL)
        >>> bins.get_projection()[3]
        0.0043103448275862068
        >>> # bins.debug("colorful horizontal")
        >>>
        >>> # vertical
        >>> bins = ColorfulProjection(arr, Projection.TYPE_VERTICAL)
        >>> bins.get_projection()[3]
        0.047008547008547008
        >>> # bins.debug("colorful vertical")
        """
        new_shape = (1, self.measurement) if self.orientation == self.TYPE_HORIZONTAL else (self.measurement, 1)
        vector = cv2.resize(self.image, new_shape, interpolation=interpolation_type).flatten()
        vector = 255 - vector
        assert len(vector) == self.measurement
        return Projection.normalize(vector)
