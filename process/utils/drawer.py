import cv2


class Drawer:
    def __init__(self):
        self.face = None
        self.frame = None

    def set_frame(self, frame):
        self.frame = frame

    def set_face(self, face):
        self.face = face

    def init(self, face, frame):
        self.set_face(face)
        self.set_frame(frame)

    def draw_all(self):
        self.draw_face_rectangle()
        self.draw_5_point_landmark()
        self.draw_full_landmark_points()

    def draw_face_rectangle(self):
        if self.face is None or self.frame is None:
            return
        box = self.face.get('bbox').astype(int)
        color = (100, 255, 0)
        cv2.rectangle(self.frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(self.frame, str(int(self.face.get('det_score') * 100)), (box[0] - 1, box[1] - 4),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)

    def draw_5_point_landmark(self):
        if self.face is None or self.frame is None:
            return
        kps = self.face.get('kps').astype(int)
        for point_index in range(kps.shape[0]):
            color = (0, 0, 255)
            if point_index == 0 or point_index == 3:
                color = (0, 255, 0)
            cv2.circle(self.frame, (kps[point_index][0], kps[point_index][1]), 2, color, 2)

    def draw_full_landmark_points(self):
        if self.face is None or self.frame is None:
            return
        landmarks = self.face.get('landmark').astype(int)
        for point_index in range(landmarks.shape[0]):
            color = (255, 0, 0)
            # cv2.circle(self.frame, (landmarks[point_index][0], landmarks[point_index][1]), 2, color, 2)
            cv2.putText(self.frame, str(point_index), (landmarks[point_index][0], landmarks[point_index][1]),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)

