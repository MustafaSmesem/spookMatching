from .arc_face import ArcFaceONNX


class FaceEncoder:
    def __init__(self, model_path):
        self.model = ArcFaceONNX(model_path)

    def encode(self, image, face):
        return self.model.get(image, face)

    def find_similarity(self, f1, f2):
        return self.model.compute_sim(f1, f2)
