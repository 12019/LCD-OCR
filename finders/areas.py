import cv2


class AreaFinder(object):
    image_object = None
    compression = 0

    def __init__(self, image_object):
        self.image_object = image_object
        self.image_object.crop_border()

    def find_regions(self, bool_bins, bins_count, measurement, offset):
        self.compression = 1.0 * measurement / bins_count

        start = False
        finish = False
        is_inside = False
        for index, i in enumerate(bool_bins):
            if i and not is_inside:
                is_inside = True
                start = index
            elif not i and is_inside:
                is_inside = False
                finish = index
                yield self._scale_range((start, finish), bins_count, offset)
        if not finish:
            yield self._scale_range((start, bins_count-1), bins_count, offset)

    def _scale_range(self, region, bins_count, offset):
        # print 'scale', region, bins_count, offset
        left = max(offset, (region[0] - 1) * self.compression)
        right = min(offset + bins_count * self.compression, region[1] + 1) * self.compression
        # print 'scaled:', left, right
        return left, right


class DigitsAreaFinder(AreaFinder):
    # """
    # >>> import os
    # >>> row_path = './img/tests/row.jpg'
    # >>> os.path.isfile(row_path)
    # True
    # >>> row = cv2.imread(row_path, 0)
    # >>> row.shape
    # (495, 1257)
    # >>> d = DigitsAreaFinder(row, 11.44, 560.56, 26.2, 1283.8)
    # >>> extracted = list(d.extract())[0]
    # >>> extracted
    # (11.44, 984.8, 26.2, 1283.8)
    # >>> row.shape[0] > extracted[1]
    # True
    # >>> row.shape[1] > extracted[3]
    # True
    # >>> l, r, t, b = extracted
    # >>> Utils.show_image(row)
    # >>> Utils.show_image(row[l:r, t:b])
    # """
    bins_count = 10
    threshold = 200

    def __init__(self, image_object):
        super(DigitsAreaFinder, self).__init__(image_object)

    def extract(self):
        self.split_into_bins((self.bins_count, 1))
        bool_bins = self.bins[0] < self.threshold
        # print self.bins
        # Utils.show_image(self.full_color)
        # print self.th.shape[1]
        regions = self.find_regions(bool_bins, self.bins_count, self.black_and_white.shape[1], self.left)

        for reg in regions:
            # result = self.black_and_white[:, reg[0]:reg[1]]
            # print reg
            # Utils.show_image(result)
            yield reg[0], reg[1], self.top, self.bottom

    def crop_border(self):
        # not needed
        pass


class RowAreasFinder(AreaFinder):
    # """
    # >>> import os
    # >>> path = one_line_lcd.jpg    >>> os.path.isfile(path)
    # True
    # >>> img = cv2.imread(path, 0)
    # >>> img.shape
    # (572, 1310)
    # >>> r = RowAreasFinder(img, 0, img.shape[0], 0, img.shape[1])
    # >>> list(r.extract())
    # [(26.2, 521.55, 26.2, 1131.3)]
    # """
    bins_count = 20
    threshold = 200

    def __init__(self, full_color, left, right, top, bottom):
        super(RowAreasFinder, self).__init__(full_color, left, right, top, bottom)
        self.extract()

    def extract(self):
        self.split_into_bins((1, self.bins_count))
        bool_bins = cv2.transpose(self.bins)[0] < self.threshold
        regions = self.find_regions(bool_bins, self.bins_count, self.full_color.shape[0], self.top)
        # regions = list(regions)[1:]
        for reg in regions:
            result = self.full_color[reg[0]:reg[1], :]
            # print 'reg', reg
            # Utils.show_image(result)
            # cv2.imwrite('img/tests/row.jpg', result)
            f = DigitsAreaFinder(self.full_color, reg[0], reg[1], 0, self.full_color.shape[1])
            for t in f.extract():
                yield t
