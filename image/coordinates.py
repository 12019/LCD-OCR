

class Coordinates:
    left = 0
    right = 0
    top = 0
    bottom = 0

    def __init__(self, left, right, top, bottom):
        self.left = 0
        self.top = 0
        self.right = right
        self.bottom = bottom

    @classmethod
    def from_ndarray(cls, a):
        return cls(a.left, a.right, a.top, a.bottom)

    def __str__(self):
        return "%s %s %s %s" % (self.left, self.right, self.top, self.bottom)

    def get_shape(self):
        return self.right - self.left, self.bottom - self.top

    def crop(self, left, right, top, bottom):
        if right:
            self.right = right + self.left
        if left:
            self.left += left
        if bottom:
            self.bottom = bottom + self.top
        if top:
            self.top += top
        assert left >= 0
        assert right >= 0
        assert top >= 0
        assert bottom >= 0
        assert left < right or left == 0
        assert top < bottom or top == 0
        assert self.left < self.right or self.left == 0
        assert self.top < self.bottom or self.top == 0

    def crop_borders(self, vertical_percent):
        """
        :param vertical_percent: border width in range from 0 to 1
        :return: void

        >>> c = Coordinates(0, 100, 200, 400)
        >>> c.get_shape()
        (100, 200)
        >>> c.crop_borders(0.1)
        >>> c.get_shape()
        (90, 190)
        """
        pixels = self.get_shape()[0] * vertical_percent
        self.left += pixels
        self.right -= pixels
        self.top += pixels
        self.bottom -= pixels
