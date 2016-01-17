import os.path

import cv2

from finders.areas import DigitsAreaFinder
from utils.utils import Utils
from quality_assurance import QualityAssurance
from areas import RowAreasFinder


class LCDFinder:
    current = 11
    img = None
    contours = []
    input_path = 'img/%d.jpg'
    output_path = 'img/lcd/%d.jpg'

    @staticmethod
    def _polygon_area(corners):
        # http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        n = len(corners)  # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2.0
        return area

    def polygon_area(self, cnt):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        return self._polygon_area(box)

    def calculate_rectangularity(self, cnt):
        polygon_area = self.polygon_area(cnt)
        contour_area = cv2.contourArea(cnt)
        return contour_area / polygon_area

    def find_large_rectangles(self, rectangularity_threshold, area_threshold):
        remaining = [cnt
                     for cnt in self.contours
                     if self.polygon_area(cnt) > area_threshold
                     and self.calculate_rectangularity(cnt) > rectangularity_threshold]
        # print len(list(remaining))
        # print [self.polygon_area(cnt) for cnt in remaining]
        # print [self.calculate_rectangularity(cnt) for cnt in remaining]
        return sorted(remaining, key=cv2.contourArea, reverse=True)

    def found(self, center, shape, angle):
        subimage = Utils.subimage_rotated(self.img, center, angle, int(shape[0]), int(shape[1]))
        output = self.output_path % self.current
        # r = cv2.imwrite(output, subimage)
        # Utils.show_image(subimage)
        # DigitsFinder(subimage)
        f = RowAreasFinder(subimage, 0, 0)
        for left, right, top, bottom in f.extract():
            print left, right, top, bottom
            Utils.show_image(subimage[left:right, top:bottom])

    def __init__(self, filename):
        self.current = int(filename[6:8])
        path = self.input_path % int(self.current)
        print '\n\n' + path
        assert os.path.isfile(path), "File not exists"

        self.img = cv2.imread(path, 0)
        # blurred_img = cv2.medianBlur(img,5)
        ret, th1 = cv2.threshold(self.img, 127, 255, cv2.THRESH_OTSU)
        im2, self.contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        QualityAssurance(self.img)

        rectangles = self.find_large_rectangles(0.9, 1E3)
        # Utils.show_contour(self.img.shape, rectangles)
        assert len(rectangles), "No LCD display found"

        center, shape, angle = cv2.minAreaRect(rectangles[0])
        self.found(center, shape, angle)
