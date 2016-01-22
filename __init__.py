import glob
from visible.photo import Photo


files = sorted(glob.glob('img/[0-9][0-9].jpg'))
# files = ["./img/22.jpg"]
print files
for f in files[:]:
    try:
        p = Photo(f)
        print p.extract_areas_with_digits(True, True)
    except AssertionError as e:
        print e
