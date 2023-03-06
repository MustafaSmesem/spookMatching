from process import FaceDetector, LandmarkDetector, AgeGenderDetector, FaceEncoder, Drawer
from process import result_to_faces, judge_side_face
import os


class ImageAnalyzer:
    def __init__(self, config):
        models_dir = config['models_dir']
        self.face_detector = FaceDetector(os.path.join(models_dir, config['det_model_file']), config['det_size'])
        self.face_encoder = FaceEncoder(os.path.join(models_dir, config['arcface_model_path']))
        self.landmark_detector = LandmarkDetector(os.path.join(models_dir, config['landmark_model_file']))
        self.age_gender_detector = AgeGenderDetector(os.path.join(models_dir, config['age_gender_model_path']))
        self.drawer = Drawer(draw_det_score=True)
        self.face_margin = config['face_margin']

    def analyze_image(self, img):
        boxes, kps = self.face_detector.auto_detect(img)
        faces = result_to_faces(boxes, kps)
        for face in faces:
            encodings = self.face_encoder.encode(img, face)
            print(encodings)
