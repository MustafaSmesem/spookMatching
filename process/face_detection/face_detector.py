from .retinaface import get_retinaface
from ..utils import Face


class FaceDetector:
    def __init__(self, model_path, det_size=(640, 640)):
        self.model = get_retinaface(model_path)
        self.det_size = det_size

    def detect(self, frame):
        boxes, kps = self.model.detect(frame, self.det_size)
        faces = []
        for i in range(len(boxes)):
            face = Face(bbox=boxes[i][0:4], kps=kps[i], det_score=boxes[i][4])
            faces.append(face)
        return faces

