

class Coordinates:
    left = 0
    right = 0
    top = 0
    bottom = 0

    def __init__(self, top, bottom, left, right):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    @classmethod
    def from_ndarray(cls, a):
        return cls(0, a.shape[0], 0, a.shape[1])

    def __repr__(self):
        return "(%d:%d, %d:%d)" % (self.top, self.bottom, self.left, self.right)

    def get_shape(self):
        assert self.right - self.left > self.bottom - self.top
        return self.bottom - self.top, self.right - self.left
        # return self.right - self.left, self.bottom - self.top

    def crop(self, coord):
        """Virtual crop.

        :param coord - instance of Coordinates object

        >>> import cv2
        >>> arr = cv2.imread("./img/tests/one_line_lcd.jpg", 0)
        >>> c = Coordinates.from_ndarray(arr)
        >>> arr.shape
        (572, 1310)
        >>> c.get_shape()  # (vertical, horizontal)
        (572, 1310)
        >>>
        >>> # first crop
        >>> c.crop(Coordinates(310, 510, 100, 1000))
        >>> split_one = arr[310:510, 100:1000]
        >>> split_one.shape
        (200, 900)
        >>> c.get_shape()
        (200, 900)
        >>> split_one.shape == c.get_shape()
        True
        >>> str(c)
        '(310:510, 100:1000)'
        >>>
        >>> # second crop
        >>> c.crop(Coordinates(100, 200, 100, 500))
        >>> c.get_shape()
        (100, 400)
        >>> arr[310:510, 100:1000][100:200, 100:500].shape
        (100, 400)
        >>> arr[310:510, 100:1000][100:200, 100:500].shape == c.get_shape()
        True
        >>> str(c)
        '(410:510, 200:600)'
        """
        if coord.right:
            self.right = coord.right + self.left
        if coord.left:
            self.left += coord.left
        if coord.bottom:
            self.bottom = coord.bottom + self.top
        if coord.top:
            self.top += coord.top
        assert coord.left >= 0
        assert coord.right >= 0
        assert coord.top >= 0
        assert coord.bottom >= 0
        assert coord.left < coord.right
        assert coord.top < coord.bottom
        assert self.left < self.right
        assert self.top < self.bottom

    def crop_borders(self, width_percent):
        """
        :param width_percent: border width in range from 0 to 1
        :return: void

        >>> c = Coordinates(200, 400, 0, 1000)
        >>> c
        (200:400, 0:1000)
        >>> c.get_shape()
        (200, 1000)
        >>> c.crop_borders(0.1)
        >>> c.get_shape()
        (160.0, 960.0)
        """
        pixels = self.get_shape()[0] * width_percent
        self.left += pixels
        self.right -= pixels
        self.top += pixels
        self.bottom -= pixels
