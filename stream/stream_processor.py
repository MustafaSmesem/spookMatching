from process import FaceDetector
from process import LandmarkDetector
from process import Drawer
import cv2
import time


class StreamProcessor:
    def __init__(self, config):
        self.face_detector = FaceDetector(config['models_dir'] + config['det_model_file'], config['det_size'])
        self.landmark_detector = LandmarkDetector(config['models_dir'] + config['landmark_model_file'])
        self.drawer = Drawer()

    def run(self):
        vid = cv2.VideoCapture(0)
        frame_number = 0
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

            start = time.time()
            faces = self.face_detector.detect(frame)
            for face in faces:
                self.landmark_detector.detect(frame, face)
                self.drawer.init(face, frame)
                self.drawer.draw_all()

            print(f'detection [%d]: %.3f sn' % (frame_number, time.time() - start))
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
