import math
import numpy as np
import os

import cv2

from croppable.croppable_image import CroppableImage
from qa.quality_assurance import QualityAssurance
from utilities.plotter import Plotter
from visible.lcd import LCD


class Photo:
    """
    >>> p = Photo("assets/img/doctests/photo.jpg")
    >>> lcd = p.get_lcd_object()
    >>> lcd
    <LCD 620x1437>
    >>> images = lcd.extract_areas_with_digits()
    >>> images
    [<CroppableImage <Rectangle 0:620, 57:1255>) out of (620, 1437)>]
    >>> lcd.debug()
    """
    current = 11
    img = subimage = None
    contours = []
    coords_list = []
    INPUT_PATH = 'img/%d.jpg'

    def __init__(self, path):
        assert os.path.isfile(path), "File not exists"
        self.img = cv2.imread(path, 0)

    @staticmethod
    def from_id(file_id):
        assert int(file_id) > 0
        path = Photo.INPUT_PATH % int(file_id)
        assert os.path.isfile(path)
        return Photo(path)

    def get_lcd_object(self):
        ret, th1 = cv2.threshold(self.img, 127, 255, cv2.THRESH_OTSU)
        im2, self.contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        QualityAssurance(self.img)

        rectangles = self.find_large_rectangles(0.9, 1E3)
        # Utils.show_contour(self.img.shape, rectangles)
        assert len(rectangles), "No LCD display found"

        center, shape, angle = cv2.minAreaRect(rectangles[0])
        self.subimage = Photo.rotate_subimage(self.img, center, angle, int(shape[0]), int(shape[1]))
        if self.subimage.shape[0] > self.subimage.shape[1]:
            self.subimage = np.rot90(self.subimage)
        return LCD(CroppableImage(self.subimage))

    @staticmethod
    def polygon_area(cnt):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)

        # http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        n = len(box)  # of box
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += box[i][0] * box[j][1]
            area -= box[j][0] * box[i][1]
        area = abs(area) / 2.0
        return area

    def calculate_rectangularity(self, cnt):
        polygon_area = Photo.polygon_area(cnt)
        contour_area = cv2.contourArea(cnt)
        return contour_area / polygon_area

    def find_large_rectangles(self, rectangularity_threshold, area_threshold):
        remaining = [cnt
                     for cnt in self.contours
                     if Photo.polygon_area(cnt) > area_threshold
                     and self.calculate_rectangularity(cnt) > rectangularity_threshold]
        # print len(list(remaining))
        # print [self.polygon_area(cnt) for cnt in remaining]
        # print [self.calculate_rectangularity(cnt) for cnt in remaining]
        return sorted(remaining, key=cv2.contourArea, reverse=True)

    @staticmethod
    def rotate_subimage(image, center, theta, width, height):
        theta *= 3.14159 / 180
        v_x = (math.cos(theta), math.sin(theta))
        v_y = (-math.sin(theta), math.cos(theta))
        s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
        s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

        mapping = np.array([[v_x[0], v_y[0], s_x],
                            [v_x[1], v_y[1], s_y]])

        return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    def debug(self, window_title="Photo"):
        assert self.coords_list and self.subimage.shape
        Plotter.outline_subimages(
            CroppableImage(self.subimage),
            self.get_lcd_object(),
            window_title=window_title
        )
