from .landmark import Landmark


class LandmarkDetector:
    def __init__(self, model_path):
        self.model = Landmark(model_path)

    def detect(self, image, face):
        return self.model.get(image, face)
