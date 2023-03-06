import numpy as np
from .data_association import associate_detections_to_trackers
from .kalman_tracker import KalmanBoxTracker


class Tracker:
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, boxes, img_size, additional_attribute_list, predict_num, faces):
        self.frame_count += 1
        # get predicted locations from existing trackers.
        tracks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(tracks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if boxes != []:
            matched, unmatched_boxes, unmatched_trks = associate_detections_to_trackers(boxes, tracks)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(boxes[d, :][0])
                    trk.face_addtional_attribute.append(additional_attribute_list[d[0]])
                    trk.face = faces[d[0]]

            # create and initialise new trackers for unmatched detections
            for i in unmatched_boxes:
                trk = KalmanBoxTracker(boxes[i, :])
                trk.face_addtional_attribute.append(additional_attribute_list[i])
                trk.face = faces[i]
                print("new Tracker: {0}".format(trk.id + 1))
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if boxes == []:
                trk.update([])
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(trk)  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update >= self.max_age or trk.predict_num >= predict_num or d[2] < 0 or d[3] < 0 or \
                    d[0] > img_size[1] or d[1] > img_size[0]:
                if len(trk.face_addtional_attribute) >= 5:
                    trk.is_deleted = True
                    print('send removed tracker {}'.format(trk.id + 1))
                print('remove tracker: {0}'.format(trk.id + 1))
                self.trackers.pop(i)
        return ret
