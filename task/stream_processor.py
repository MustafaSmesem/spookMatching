from process import FaceDetector, LandmarkDetector, AgeGenderDetector, FaceEncoder, Drawer
from process import result_to_faces, judge_side_face
from tracker.tracker import Tracker
import cv2
import time
import os
import numpy as np


class StreamProcessor:
    def __init__(self, config):
        models_dir = config['models_dir']
        self.face_detector = FaceDetector(os.path.join(models_dir, config['det_model_file']), config['det_size'])
        self.face_encoder = FaceEncoder(os.path.join(models_dir, config['arcface_model_path']))
        self.landmark_detector = LandmarkDetector(os.path.join(models_dir, config['landmark_model_file']))
        self.age_gender_detector = AgeGenderDetector(os.path.join(models_dir, config['age_gender_model_path']))
        self.drawer = Drawer(draw_det_score=True)
        self.tracker = Tracker(max_age=config['tracker_max_age'])
        self.detect_interval = config['detect_interval']
        self.colors = np.random.rand(32, 3)
        self.face_score_threshold = 0.5
        self.face_margin = config['face_margin']

    def run(self):
        vid = cv2.VideoCapture(0)
        frame_number = 0
        count = 3
        ret, frame = None, None
        while count:
            ret, frame = vid.read()
            count -= 1
        if not (ret | frame).any():
            print('cannot read task.. | process canceled')
            return

        img_size = np.asarray(frame.shape)[0:2]
        while True:

            # Capture the video frame
            # by frame
            ret, frame = vid.read()
            if not ret:
                print('ret is none')
                break
            if frame is None:
                print('frame is empty')
                break

            final_faces = []
            additional_attribute_list = []
            faces = []

            start = time.time()
            if frame_number % self.detect_interval == 0:
                boxes, kps = self.face_detector.detect(frame)
                face_sums = boxes.shape[0]
                if face_sums > 0:
                    face_list = []
                    for i, item in enumerate(boxes):
                        score = round(boxes[i, 4], 6)
                        if score > self.face_score_threshold:
                            det = np.squeeze(boxes[i, 0:4])

                            # face rectangle
                            det[0] = np.maximum(det[0] - self.face_margin, 0)
                            det[1] = np.maximum(det[1] - self.face_margin, 0)
                            det[2] = np.minimum(det[2] + self.face_margin, img_size[1])
                            det[3] = np.minimum(det[3] + self.face_margin, img_size[0])
                            face_list.append(item)

                            # face cropped
                            bb = np.array(det, dtype=np.int32)
                            squeeze_points = np.squeeze(kps[i, :])
                            cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()
                            dist_rate, high_ratio_variance, width_rate = judge_side_face(np.array(squeeze_points))

                            item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate]
                            additional_attribute_list.append(item_list)

                    final_faces = np.array(face_list)
                faces = result_to_faces(boxes, kps)
            trackers = self.tracker.update(final_faces, img_size, additional_attribute_list,
                                           self.detect_interval, faces)

            for trk in trackers:
                d = (trk.get_state()).astype(np.int32)
                trk_id = trk.id
                trk_color = self.colors[trk_id % 32, :] * 255
                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), trk_color, 3)
                cv2.putText(frame, 'TRK ID : %d' % trk_id, (d[0] + 10, d[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            trk_color, 2)
                if not trk.is_recognized:
                    print('not recognized')
                    trk.is_recognized = True

                if trk.is_deleted:
                    print('trk deleted')
                face = trk.face
                # self.face_encoder.encode(frame, face)
                self.landmark_detector.detect(frame, face)
                # self.age_gender_detector.detect(frame, face)
                self.drawer.init(face, frame)
                self.drawer.draw_full_landmark_mask(trk_color, trk_color * 0.7)

            # print(f'detection [%d]: %.3f sn' % (frame_number, time.time() - start))
            cv2.putText(frame, str(frame_number), (25, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 1)
            # Display the resulting frame
            cv2.imshow('frame', frame)

            # the 'q' button is set as the
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_number += 1

        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
