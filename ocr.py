import cv2
import pytesseract
from PIL import Image

print(pytesseract.image_to_string(Image.open('assets/img/doctests/photo.jpg'), lang="eng", config="digits"))


