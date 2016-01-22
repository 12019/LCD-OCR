import numpy as np
import cv2


class DigitsSharpener:
    img = None

    def __init__(self, image):
        self.img = image

    def __read(self):
        self.img = cv2.medianBlur(self.img, 25)

        ret, th11 = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        ret, th121 = cv2.threshold(self.img, 127, 255, cv2.THRESH_TRUNC | cv2.THRESH_OTSU)  #
        ret, th122 = cv2.threshold(self.img, 150, 255, cv2.THRESH_TRUNC)  #
        ret, th131 = cv2.threshold(self.img, 127, 255, cv2.THRESH_TOZERO_INV | cv2.THRESH_OTSU)
        ret, th132 = cv2.threshold(self.img, 90, 100, cv2.THRESH_TOZERO_INV)
        ret, th14 = cv2.threshold(self.img, 127, 255, cv2.THRESH_OTSU)
        # adaptive: border detection issue
        th2 = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 551, 10)
        th3 = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 151, 10)
        return th132

    def __auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged visible
        return edged