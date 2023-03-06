from .age_gender import Attribute


class AgeGenderDetector:
    def __init__(self, model_path):
        self.model = Attribute(model_path)

    def detect(self, frame, face):
        return self.model.get(frame, face)
