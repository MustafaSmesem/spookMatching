from retinaface import get_retinaface


class FaceDetector:
    def __init__(self, model_path, config={}):
        self.model = get_retinaface(model_path)
