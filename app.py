from task.stream_processor import StreamProcessor
import os
from process import FaceDetector, Face, Drawer, FaceEncoder
import cv2
from album.bulk_album_creator import BulkAlbumCreator

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    config = {
        'models_dir': os.path.join(ROOT_DIR, 'models'),
        'det_model_file': 'scrfd_10g_bnkps.onnx',
        'det_size': (640, 640),
        'landmark_model_file': '2d106det.onnx',
        'age_gender_model_path': 'genderage.onnx',
        'arcface_model_path': 'glintr100.onnx',
        'detect_interval': 1,
        'tracker_max_age': 2,
        'face_margin': 35
    }

    # streamer = StreamProcessor(config)
    # streamer.run()
    # face_detector = FaceDetector(os.path.join(config['models_dir'], config['det_model_file']), config['det_size'])
    # face_encoder = FaceEncoder(os.path.join(config['models_dir'], config['arcface_model_path']))
    #
    # img1 = cv2.imread(os.path.join(ROOT_DIR, 'static/test_imgs/img_001.jpeg'))
    # img2 = cv2.imread(os.path.join(ROOT_DIR, 'static/test_imgs/img_002.jpeg'))
    # boxes1, kpss1 = face_detector.auto_detect(img1)
    # boxes2, kpss2 = face_detector.auto_detect(img2)
    #
    # face1 = Face(bbox=boxes1[0][0:4], kps=kpss1[0], det_score=boxes1[0][4])
    # face2 = Face(bbox=boxes2[0][0:4], kps=kpss2[0], det_score=boxes2[0][4])
    #
    # f1 = face_encoder.encode(img1, face1)
    # f2 = face_encoder.encode(img2, face2)

    # sim = face_encoder.find_similarity(f1, f2)
    # print(sim)

    album_creator = BulkAlbumCreator("/Users/arpanet_mustafa/input/test.txt",
                                     "/Users/arpanet_mustafa/output/test/mustafa", config)

    file = open('/Users/arpanet_mustafa/output/test/mustafa/test.txt', 'r')
    lines = file.readlines()
    i = 0
    for line in lines:
        i += 1
        print(f'{i} -> {line}')
