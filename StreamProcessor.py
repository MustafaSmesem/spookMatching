import cv2
from FaceAnalyzer import FaceAnalyzer


class StreamProcessor:
    def __init__(self, stream_path, **kwargs):
        self.face_analyzer = FaceAnalyzer("/home/spookArc/")
        self.stream_path = stream_path

    def run(self):
        video_cap = cv2.VideoCapture(self.stream_path)
        frame_number = 0

        while True:
            # Grab a single frame of video
            ret, frame = video_cap.read()
            frame_number += 1
            faces = self.face_analyzer.analyze(frame)