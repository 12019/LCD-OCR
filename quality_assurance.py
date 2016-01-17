class QualityAssurance:
    img = None  # numpy array

    def __init__(self, image):
        self.img = image

    def get_glare_percent(self):
        zeros = sum(sum(self.img == 255))
        pixels = self.img.shape[0] * self.img.shape[1]
        return 1.0 * zeros / pixels * 100
