import cv2


class Drawer:
    def __init__(self, draw_det_score=False):
        self.face = None
        self.frame = None
        self.draw_det_score = draw_det_score

    def set_frame(self, frame):
        self.frame = frame

    def set_face(self, face):
        self.face = face

    def init(self, face, frame):
        self.set_face(face)
        self.set_frame(frame)

    def draw_all(self):
        self.draw_face_rectangle()
        self.draw_full_landmark_mask()
        self.draw_age_gender()

    def draw_face_rectangle(self, color=(100, 255, 0)):
        if self.face is None or self.frame is None:
            return
        box = self.face.bbox.astype(int)
        cv2.rectangle(self.frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        if self.draw_det_score:
            cv2.putText(self.frame, f'%.2f %%' % (self.face.det_score * 100), (box[0] - 1, box[3] + 35),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (150, 255, 50), 1)

    def draw_5_point_landmark(self, color1=(0, 0, 255), color2=(0, 255, 0)):
        if self.face is None or self.frame is None:
            return
        kps = self.face.kps.astype(int)
        for point_index in range(kps.shape[0]):
            color = color1
            if point_index == 0 or point_index == 3:
                color = color2
            cv2.circle(self.frame, (kps[point_index][0], kps[point_index][1]), 2, color, 2)

    def draw_full_landmark_points(self, color=(200, 255, 0)):
        if self.face is None or self.frame is None or self.face.landmark is None:
            return []
        landmarks = self.face.landmark.astype(int)
        for point_index in range(landmarks.shape[0]):
            cv2.circle(self.frame, (landmarks[point_index][0], landmarks[point_index][1]), 3, color, 2)
        return landmarks

    def draw_line(self, point1, point2, color=(150, 255, 100)):
        thickness = 1
        cv2.line(self.frame, (point1[0], point1[1]), (point2[0], point2[1]), color, thickness)

    def draw_full_landmark_mask(self, color=(150, 255, 100), points_color=(200, 255, 0)):
        landmarks = self.draw_full_landmark_points(points_color)
        if len(landmarks) == 106:
            # face contour
            self.draw_line(landmarks[1], landmarks[9], color)
            self.draw_line(landmarks[9], landmarks[10], color)
            self.draw_line(landmarks[10], landmarks[11], color)
            self.draw_line(landmarks[11], landmarks[12], color)
            self.draw_line(landmarks[12], landmarks[13], color)
            self.draw_line(landmarks[13], landmarks[14], color)
            self.draw_line(landmarks[14], landmarks[15], color)
            self.draw_line(landmarks[15], landmarks[16], color)
            self.draw_line(landmarks[16], landmarks[2], color)
            self.draw_line(landmarks[2], landmarks[3], color)
            self.draw_line(landmarks[3], landmarks[4], color)
            self.draw_line(landmarks[4], landmarks[5], color)
            self.draw_line(landmarks[5], landmarks[6], color)
            self.draw_line(landmarks[6], landmarks[7], color)
            self.draw_line(landmarks[7], landmarks[8], color)
            self.draw_line(landmarks[8], landmarks[0], color)
            self.draw_line(landmarks[0], landmarks[24], color)
            self.draw_line(landmarks[24], landmarks[23], color)
            self.draw_line(landmarks[23], landmarks[22], color)
            self.draw_line(landmarks[22], landmarks[21], color)
            self.draw_line(landmarks[21], landmarks[20], color)
            self.draw_line(landmarks[20], landmarks[19], color)
            self.draw_line(landmarks[19], landmarks[18], color)
            self.draw_line(landmarks[18], landmarks[32], color)
            self.draw_line(landmarks[32], landmarks[31], color)
            self.draw_line(landmarks[31], landmarks[30], color)
            self.draw_line(landmarks[30], landmarks[29], color)
            self.draw_line(landmarks[29], landmarks[28], color)
            self.draw_line(landmarks[28], landmarks[27], color)
            self.draw_line(landmarks[27], landmarks[26], color)
            self.draw_line(landmarks[26], landmarks[25], color)
            self.draw_line(landmarks[25], landmarks[17], color)
            self.draw_line(landmarks[17], landmarks[101], color)
            self.draw_line(landmarks[101], landmarks[105], color)
            self.draw_line(landmarks[105], landmarks[104], color)
            self.draw_line(landmarks[104], landmarks[103], color)
            self.draw_line(landmarks[103], landmarks[102], color)
            self.draw_line(landmarks[102], landmarks[50], color)
            self.draw_line(landmarks[50], landmarks[51], color)
            self.draw_line(landmarks[51], landmarks[49], color)
            self.draw_line(landmarks[49], landmarks[48], color)
            self.draw_line(landmarks[48], landmarks[43], color)
            self.draw_line(landmarks[43], landmarks[1], color)

            self.draw_line(landmarks[53], landmarks[0], color)
            self.draw_line(landmarks[53], landmarks[8], color)
            self.draw_line(landmarks[53], landmarks[24], color)
            self.draw_line(landmarks[56], landmarks[6], color)
            self.draw_line(landmarks[56], landmarks[7], color)
            self.draw_line(landmarks[59], landmarks[22], color)
            self.draw_line(landmarks[59], landmarks[23], color)
            self.draw_line(landmarks[58], landmarks[21], color)
            self.draw_line(landmarks[58], landmarks[20], color)
            self.draw_line(landmarks[61], landmarks[19], color)
            self.draw_line(landmarks[61], landmarks[18], color)
            self.draw_line(landmarks[52], landmarks[2], color)
            self.draw_line(landmarks[52], landmarks[3], color)
            self.draw_line(landmarks[55], landmarks[4], color)
            self.draw_line(landmarks[55], landmarks[5], color)

            # mouse
            self.draw_line(landmarks[52], landmarks[55], color)
            self.draw_line(landmarks[55], landmarks[56], color)
            self.draw_line(landmarks[56], landmarks[53], color)
            self.draw_line(landmarks[53], landmarks[59], color)
            self.draw_line(landmarks[59], landmarks[58], color)
            self.draw_line(landmarks[58], landmarks[61], color)
            self.draw_line(landmarks[61], landmarks[68], color)
            self.draw_line(landmarks[68], landmarks[67], color)
            self.draw_line(landmarks[67], landmarks[71], color)
            self.draw_line(landmarks[71], landmarks[63], color)
            self.draw_line(landmarks[63], landmarks[64], color)
            self.draw_line(landmarks[64], landmarks[52], color)

            self.draw_line(landmarks[65], landmarks[66], color)
            self.draw_line(landmarks[66], landmarks[62], color)
            self.draw_line(landmarks[62], landmarks[70], color)
            self.draw_line(landmarks[70], landmarks[69], color)
            self.draw_line(landmarks[69], landmarks[57], color)
            self.draw_line(landmarks[57], landmarks[60], color)
            self.draw_line(landmarks[60], landmarks[54], color)
            self.draw_line(landmarks[54], landmarks[65], color)

            self.draw_line(landmarks[52], landmarks[65], color)
            self.draw_line(landmarks[69], landmarks[61], color)
            self.draw_line(landmarks[55], landmarks[54], color)
            self.draw_line(landmarks[57], landmarks[58], color)
            self.draw_line(landmarks[66], landmarks[64], color)
            self.draw_line(landmarks[70], landmarks[68], color)
            self.draw_line(landmarks[60], landmarks[56], color)
            self.draw_line(landmarks[60], landmarks[53], color)
            self.draw_line(landmarks[60], landmarks[59], color)
            self.draw_line(landmarks[62], landmarks[71], color)
            self.draw_line(landmarks[62], landmarks[63], color)
            self.draw_line(landmarks[62], landmarks[67], color)

            # noise
            self.draw_line(landmarks[102], landmarks[97], color)
            self.draw_line(landmarks[97], landmarks[81], color)
            self.draw_line(landmarks[81], landmarks[82], color)
            self.draw_line(landmarks[82], landmarks[83], color)
            self.draw_line(landmarks[83], landmarks[84], color)
            self.draw_line(landmarks[84], landmarks[85], color)
            self.draw_line(landmarks[85], landmarks[80], color)
            self.draw_line(landmarks[80], landmarks[79], color)
            self.draw_line(landmarks[79], landmarks[78], color)
            self.draw_line(landmarks[78], landmarks[77], color)
            self.draw_line(landmarks[77], landmarks[76], color)
            self.draw_line(landmarks[76], landmarks[75], color)
            self.draw_line(landmarks[75], landmarks[46], color)
            self.draw_line(landmarks[46], landmarks[50], color)

            self.draw_line(landmarks[86], landmarks[80], color)
            self.draw_line(landmarks[86], landmarks[82], color)
            self.draw_line(landmarks[86], landmarks[83], color)
            self.draw_line(landmarks[86], landmarks[84], color)
            self.draw_line(landmarks[86], landmarks[85], color)
            self.draw_line(landmarks[86], landmarks[76], color)
            self.draw_line(landmarks[86], landmarks[77], color)
            self.draw_line(landmarks[86], landmarks[78], color)
            self.draw_line(landmarks[86], landmarks[79], color)

            self.draw_line(landmarks[72], landmarks[97], color)
            self.draw_line(landmarks[72], landmarks[81], color)
            self.draw_line(landmarks[72], landmarks[46], color)
            self.draw_line(landmarks[72], landmarks[75], color)
            self.draw_line(landmarks[72], landmarks[102], color)
            self.draw_line(landmarks[72], landmarks[50], color)

            self.draw_line(landmarks[73], landmarks[81], color)
            self.draw_line(landmarks[73], landmarks[75], color)
            self.draw_line(landmarks[73], landmarks[74], color)
            self.draw_line(landmarks[82], landmarks[74], color)
            self.draw_line(landmarks[76], landmarks[74], color)

            # eyebrow
            self.draw_line(landmarks[43], landmarks[44], color)
            self.draw_line(landmarks[44], landmarks[45], color)
            self.draw_line(landmarks[45], landmarks[47], color)
            self.draw_line(landmarks[47], landmarks[46], color)

            self.draw_line(landmarks[97], landmarks[98], color)
            self.draw_line(landmarks[98], landmarks[99], color)
            self.draw_line(landmarks[99], landmarks[100], color)
            self.draw_line(landmarks[100], landmarks[101], color)

            self.draw_line(landmarks[44], landmarks[48], color)
            self.draw_line(landmarks[44], landmarks[49], color)
            self.draw_line(landmarks[45], landmarks[48], color)
            self.draw_line(landmarks[45], landmarks[49], color)
            self.draw_line(landmarks[45], landmarks[51], color)
            self.draw_line(landmarks[47], landmarks[49], color)
            self.draw_line(landmarks[47], landmarks[50], color)
            self.draw_line(landmarks[47], landmarks[51], color)

            self.draw_line(landmarks[100], landmarks[104], color)
            self.draw_line(landmarks[100], landmarks[105], color)
            self.draw_line(landmarks[99], landmarks[103], color)
            self.draw_line(landmarks[99], landmarks[104], color)
            self.draw_line(landmarks[99], landmarks[105], color)
            self.draw_line(landmarks[98], landmarks[102], color)
            self.draw_line(landmarks[98], landmarks[103], color)
            self.draw_line(landmarks[98], landmarks[104], color)

            # eyes
            self.draw_line(landmarks[35], landmarks[36], color)
            self.draw_line(landmarks[36], landmarks[33], color)
            self.draw_line(landmarks[33], landmarks[37], color)
            self.draw_line(landmarks[37], landmarks[39], color)
            self.draw_line(landmarks[39], landmarks[42], color)
            self.draw_line(landmarks[42], landmarks[40], color)
            self.draw_line(landmarks[40], landmarks[41], color)
            self.draw_line(landmarks[41], landmarks[35], color)

            self.draw_line(landmarks[89], landmarks[90], color)
            self.draw_line(landmarks[90], landmarks[87], color)
            self.draw_line(landmarks[87], landmarks[91], color)
            self.draw_line(landmarks[91], landmarks[93], color)
            self.draw_line(landmarks[93], landmarks[96], color)
            self.draw_line(landmarks[96], landmarks[94], color)
            self.draw_line(landmarks[94], landmarks[95], color)
            self.draw_line(landmarks[95], landmarks[89], color)

            # general lines
            self.draw_line(landmarks[77], landmarks[33], color)
            self.draw_line(landmarks[77], landmarks[52], color)
            self.draw_line(landmarks[83], landmarks[87], color)
            self.draw_line(landmarks[83], landmarks[61], color)

            self.draw_line(landmarks[35], landmarks[43], color)
            self.draw_line(landmarks[35], landmarks[1], color)
            self.draw_line(landmarks[35], landmarks[9], color)
            self.draw_line(landmarks[35], landmarks[10], color)
            self.draw_line(landmarks[35], landmarks[52], color)
            self.draw_line(landmarks[41], landmarks[44], color)
            self.draw_line(landmarks[40], landmarks[45], color)
            self.draw_line(landmarks[47], landmarks[42], color)

            self.draw_line(landmarks[93], landmarks[101], color)
            self.draw_line(landmarks[93], landmarks[17], color)
            self.draw_line(landmarks[93], landmarks[25], color)
            self.draw_line(landmarks[93], landmarks[26], color)
            self.draw_line(landmarks[93], landmarks[61], color)
            self.draw_line(landmarks[95], landmarks[98], color)
            self.draw_line(landmarks[94], landmarks[99], color)
            self.draw_line(landmarks[96], landmarks[100], color)

            # self.draw_line(landmarks[77], landmarks[16])
            # self.draw_line(landmarks[77], landmarks[15])
            # self.draw_line(landmarks[77], landmarks[14])

            # self.draw_line(landmarks[83], landmarks[32])
            # self.draw_line(landmarks[83], landmarks[31])
            # self.draw_line(landmarks[83], landmarks[30])

            self.draw_line(landmarks[39], landmarks[46], color)
            self.draw_line(landmarks[39], landmarks[76], color)
            self.draw_line(landmarks[89], landmarks[97], color)
            self.draw_line(landmarks[89], landmarks[82], color)

            self.draw_line(landmarks[78], landmarks[64], color)
            self.draw_line(landmarks[79], landmarks[63], color)
            self.draw_line(landmarks[80], landmarks[71], color)
            self.draw_line(landmarks[85], landmarks[67], color)
            self.draw_line(landmarks[84], landmarks[68], color)

    def draw_age_gender(self, color=(150, 255, 50)):
        if self.face is None or self.frame is None or self.face.age is None or self.face.gender:
            return
        box = self.face.bbox.astype(int)
        cv2.putText(self.frame, f'%d old, %s' % (self.face.age, 'Male' if self.face.gender == 1 else 'Female'),
                    (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1)
