

class Rectangle:
    """
    Modelled after http://docs.opencv.org/2.4/modules/core/doc/basic_structures.html#rect
    """
    left = 0
    width = 0
    top = 0
    height = 0

    def __init__(self, top, height, left, width):
        self.left = left
        self.width = width
        self.top = top
        self.height = height

    @classmethod
    def from_ndarray(cls, a):
        return cls(0, a.shape[0], 0, a.shape[1])

    def __repr__(self):
        return "(%d:%d, %d:%d)" % (self.top, self.height, self.left, self.width)

    def get_shape(self):
        assert self.width - self.left > self.height - self.top
        return self.height - self.top, self.width - self.left
        # return self.width - self.left, self.height - self.top

    def crop(self, coord):
        """Virtual crop.

        :param coord - instance of Coordinates object

        >>> import cv2
        >>> arr = cv2.imread("assets/img/doctests/single_line_lcd.jpg", 0)
        >>> c = Rectangle.from_ndarray(arr)
        >>> arr.shape
        (572, 1310)
        >>> c.get_shape()  # (vertical, horizontal)
        (572, 1310)
        >>>
        >>> # first crop
        >>> c.crop(Rectangle(310, 510, 100, 1000))
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
        >>> c.crop(Rectangle(100, 200, 100, 500))
        >>> c.get_shape()
        (100, 400)
        >>> arr[310:510, 100:1000][100:200, 100:500].shape
        (100, 400)
        >>> arr[310:510, 100:1000][100:200, 100:500].shape == c.get_shape()
        True
        >>> str(c)
        '(410:510, 200:600)'
        """
        if coord.width:
            self.width = coord.width + self.left
        if coord.left:
            self.left += coord.left
        if coord.height:
            self.height = coord.height + self.top
        if coord.top:
            self.top += coord.top
        assert coord.left >= 0
        assert coord.width >= 0
        assert coord.top >= 0
        assert coord.height >= 0
        assert coord.left < coord.width
        assert coord.top < coord.height
        assert self.left < self.width
        assert self.top < self.height

    def crop_borders(self, width_percent):
        """
        :param width_percent: border width in range from 0 to 1
        :return: void

        >>> c = Rectangle(200, 400, 0, 1000)
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
        self.width -= pixels
        self.top += pixels
        self.height -= pixels
