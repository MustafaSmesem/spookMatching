import os

import numpy as np

from process import FaceEncoder
from process import FaceDetector
from process import LandmarkDetector
from process import Face
from process import Drawer
import cv2


class BulkAlbumCreator:
    def __init__(self, input_file, output_folder, config):
        self.input_file = input_file
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        models_dir = config['models_dir']
        self.face_detector = FaceDetector(os.path.join(models_dir, config['det_model_file']), config['det_size'])
        self.face_encoder = FaceEncoder(os.path.join(models_dir, config['arcface_model_path']))
        self.landmark_detector = LandmarkDetector(os.path.join(models_dir, config['landmark_model_file']))
        self.drawer = Drawer()
        self.parse_file()

    def analyse_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f'image[{image_path}] cannot be open')
            return [], []
        boxes, kps = self.face_detector.auto_detect(img)
        if len(boxes) != 1:
            print(f'there {len(boxes)} face must be 1')
            return [], []

        face = Face(bbox=boxes[0][0:4], kps=kps[0], det_score=boxes[0][4])
        encodings = self.face_encoder.encode(img, face)
        landmarks = self.landmark_detector.detect(img, face)
        x, y, w, h = boxes[0].astype(int)[:4]
        cv2.imwrite(image_path[:image_path.rfind('.')] + '_face.jpg', img[y:h, x:w])
        return encodings, landmarks

    def parse_file(self):
        file = open(self.input_file, 'r')
        lines = file.readlines()
        encodings_file = open(os.path.join(self.output_folder, self.input_file[self.input_file.rfind('/')+1:]), 'a')

        count = 0
        for line in lines:
            count += 1
            image_data = line.split(',')
            if len(image_data) != 4:
                continue
            enc, lndmrk = self.analyse_image(image_data[2])
            if len(enc) == 512:
                encodings_file.write(image_data[0] + '|' + image_data[1] + '|' + str(enc.tolist()) + "|" + str(lndmrk.tolist()) + '\n')
        encodings_file.close()
        file.close()


