import numpy as np


def judge_side_face(facial_landmarks):
    wide_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[1])
    high_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[3])
    dist_rate = high_dist / wide_dist

    # cal std
    vec_a = facial_landmarks[0] - facial_landmarks[2]
    vec_b = facial_landmarks[1] - facial_landmarks[2]
    vec_c = facial_landmarks[3] - facial_landmarks[2]
    vec_d = facial_landmarks[4] - facial_landmarks[2]
    dist_a = np.linalg.norm(vec_a)
    dist_b = np.linalg.norm(vec_b)
    dist_c = np.linalg.norm(vec_c)
    dist_d = np.linalg.norm(vec_d)

    # cal rate
    high_rate = dist_a / dist_c
    width_rate = dist_c / dist_d
    high_ratio_variance = np.fabs(high_rate - 1.1)
    width_ratio_variance = np.fabs(width_rate - 1)

    return dist_rate, high_ratio_variance, width_ratio_variance


def add_face_margin(det, face_margin, img_size):
    det = np.squeeze(det[0:4])
    # face rectangle
    det[0] = np.maximum(det[0] - face_margin, 0)
    det[1] = np.maximum(det[1] - face_margin, 0)
    det[2] = np.minimum(det[2] + face_margin, img_size[1])
    det[3] = np.minimum(det[3] + face_margin, img_size[0])