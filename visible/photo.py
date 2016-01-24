import numpy as np
import os

import cv2
from visible.outline import Outline
from visible.lcd import LCD
from qa.quality_assurance import QualityAssurance
from utils.utils import Utils
from croppable_image import CroppableImage


class Photo:
    """
    >>> p = Photo("img/tests/photo.jpg", 99)
    >>> list(p.extract_areas_with_digits(True, True))
    [(62:558, 320:1247)]
    >>> # p.debug()
    """
    current = 11
    img = subimage = None
    contours = []
    coords_list = []
    input_path = 'img/%d.jpg'
    digits_path = 'img/digits/%d-%d.jpg'
    lcd_path = 'img/lcd/%d.jpg'

    def __init__(self, path, file_id=0):
        assert os.path.isfile(path), "File not exists"
        self.current = file_id or int(path[4:6])
        assert not file_id or self.current, "Wrong input file"
        self.img = cv2.imread(path, 0)

    def extract_areas_with_digits(self, write_lcd=False, write_digits=False):
        ret, th1 = cv2.threshold(self.img, 127, 255, cv2.THRESH_OTSU)
        im2, self.contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        QualityAssurance(self.img)

        rectangles = self.find_large_rectangles(0.9, 1E3)
        # Utils.show_contour(self.img.shape, rectangles)
        assert len(rectangles), "No LCD display found"

        center, shape, angle = cv2.minAreaRect(rectangles[0])
        self.subimage = Utils.subimage_rotated(self.img, center, angle, int(shape[0]), int(shape[1]))
        if self.subimage.shape[0] > self.subimage.shape[1]:
            self.subimage = np.rot90(self.subimage)
        if write_lcd:
            self.write_lcd(self.subimage)
        f = LCD(self.subimage)
        self.coords_list = f.extract_digits_area()
        if write_digits:
            self.coords_list = list(self.coords_list)
            self.write_digits(self.subimage, self.coords_list)
        return self.coords_list

    def write_lcd(self, ndarray):
        out = self.lcd_path % self.current
        cv2.imwrite(out, ndarray)

    def write_digits(self, ndarray, coords_list):
        for i, coords in enumerate(coords_list):
            ci = CroppableImage(ndarray, coords)
            out = self.digits_path % (self.current, i + 1)
            cv2.imwrite(out, ci.get_ndarray())

    def debug(self):
        assert self.coords_list and self.subimage.shape
        # http://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
        o = Outline(self.subimage.copy(), self.extract_areas_with_digits())
        o.show()

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
