import numpy as np
from data_association import associate_detections_to_trackers
from kalman_tracker import KalmanBoxTracker
import cv2


global fid, colours
colours = np.random.rand(32, 3)
fid = {}
idCounter = {}


def send_result(msg):
    print(f'send result %s' % msg)

class Tracker:

    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        


    def update(self, dets, img_size, root_dic, addtional_attribute_list, predict_num, frame):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE:as in practical realtime MOT, the detector doesn't run on every single frame
        """
        result = {}
        global idCounter, fid
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()  # kalman predict ,very fast ,<1ms
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if dets != []:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])
                    trk.face_addtional_attribute.append(addtional_attribute_list[d[0]])

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                trk.face_addtional_attribute.append(addtional_attribute_list[i])
                # logger.info("new Tracker: {0}".format(trk.id + 1))
                image = addtional_attribute_list[i][0]
                f, similarity = 'faceMatcher(image)'
                if similarity > 0.9:
                    fid[trk.id + 1] = f
                idCounter[trk.id + 1] = 1

                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([])
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update >= self.max_age or trk.predict_num >= predict_num or d[2] < 0 or d[3] < 0 or d[0] > img_size[1] or d[1] > img_size[0]:
                self.trackers.pop(i)
        if len(ret) > 0:
            t = np.concatenate(ret)
            for d in t:
                d = d.astype(np.int32)
                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
                if d[4] in fid:
                    cv2.putText(frame, fid[d[4]], (d[0] + 10, d[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colours[d[4] % 32, :] * 255, 2)
                else:
                    if d[4] in idCounter:
                        if idCounter[d[4]] %10 == 0:
                            cropped = frame[d[1]:d[3], d[0]:d[2], :].copy()
                            f, similarity = 'faceMatcher(cropped)'
                            if similarity > 0.9:
                                fid[d[4]] = f
                        else:
                            idCounter[d[4]] += 1

        send_result('result')
