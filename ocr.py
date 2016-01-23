import pytesseract
from PIL import Image

print(pytesseract.image_to_string(Image.open('img/tests/photo.jpg'), lang="eng", config="digits"))
