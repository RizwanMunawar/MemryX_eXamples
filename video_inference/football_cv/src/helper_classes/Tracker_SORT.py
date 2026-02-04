"""
Copyright (c) 2024 MemryX Inc.


GPL v3 License

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.

"""

import numpy as np

###################################################################################################
###########################Tracking Code Start#####################################################
###################################################################################################

def linear_assignment(cost_matrix): #HungarianAlg
    from scipy.optimize import linear_sum_assignment
    x,y = linear_sum_assignment(cost_matrix)

    return np.array(list(zip(x,y)))

def iou_batch(bb_test, bb_gt):
    
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

    return(o)

def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    
    r = w / float(h)

    return np.array([x, y, s, r]).reshape((4, 1))

###################################################################################################
from copy import deepcopy
class KalmanFilter(object):
    def __init__(self):
        self.x = [[0],[0],[0],[0],[0],[0],[0]]     # np.zeros((7, 1)) # state

        self.F = np.array([[1,0,0,0,1,0,0],
                           [0,1,0,0,0,1,0],
                           [0,0,1,0,0,0,1],
                           [0,0,0,1,0,0,0],
                           [0,0,0,0,1,0,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,1]])
        
        self.P = np.array([[10.,0.,0.,0.,0.,0.,0.],
                           [0.,10.,0.,0.,0.,0.,0.],
                           [0.,0.,10.,0.,0.,0.,0.],
                           [0.,0.,0.,10.,0.,0.,0.],
                           [0.,0.,0.,0.,10000.,0.,0.],
                           [0.,0.,0.,0.,0.,10000.,0.],
                           [0.,0.,0.,0.,0.,0.,10000.]])

        self.Q = np.array([[1.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,0.,0.,0.,0.,0.],
                           [0.,0.,1.,0.,0.,0.,0.],
                           [0.,0.,0.,1.,0.,0.,0.],
                           [0.,0.,0.,0.,0.5,0.,0.],
                           [0.,0.,0.,0.,0.,0.5,0.],
                           [0.,0.,0.,0.,0.,0.,0.25]])

        self.z = np.array([[None], # np.array([[None]*self.dim_z]).T
                           [None],
                           [None],
                           [None]]) 

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        if z is None:
            self.z = np.array([[None]*4]).T
            return

        R = np.array([[1.,0.,0.,0.],
                      [0.,1.,0.,0.],
                      [0.,0.,10.,0.],
                      [0.,0.,0.,10.]]) 

        H = np.array([[1,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0],
                      [0,0,1,0,0,0,0],
                      [0,0,0,1,0,0,0]])

        z = np.atleast_2d(z)

        y = z - np.dot(H, self.x)

        PHT = np.dot(self.P, H.T)

        S = np.dot(H, PHT) + R
        SI = np.linalg.inv(S)

        K = np.dot(PHT, SI)

        self.x = self.x + np.dot(K, y)

        I_KH = np.eye(7) - np.dot(K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, R), K.T)

        self.z = deepcopy(z)

###################################################################################################

class KalmanBoxTracker(object):
    
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter()
        self.kf.x[:4] = convert_bbox_to_z(bbox) # STATE VECTOR
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hit_streak = 0

        
    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
    
    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    
    def get_state(self):
        arr_u_dot = np.expand_dims(self.kf.x[4],0)
        arr_v_dot = np.expand_dims(self.kf.x[5],0)
        arr_s_dot = np.expand_dims(self.kf.x[6],0)

        return np.concatenate((convert_x_to_bbox(self.kf.x), [[0]], arr_u_dot, arr_v_dot, arr_s_dot), axis=1)
    

def associate_detections_to_trackers(detections, trackers):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > 0.2).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() ==1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]< 0.2):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self, max_age=5):
        self.trackers = []
        self.frame_count = 0
        self.max_age = max_age
        
    def update(self, dets= np.empty((0,6))):
        self.frame_count += 1
        
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)
        
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            
        for i in unmatched_dets:
            trk = KalmanBoxTracker(np.hstack((dets[i,:], np.array([0]))))
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= 2 or self.frame_count <= 2):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) #+1'd because MOT benchmark requires positive value
            i -= 1

            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0,6))



###################################################################################################
#####################################Tracking Code End#############################################
###################################################################################################

class SortTrackerPipeline:
    """
    Encapsulates SORT tracking + class/conf association.
    Inference code should treat this as a black box.
    """

    def __init__(self, max_age=60, iou_threshold=0.5):
        self.tracker = Sort(max_age=max_age)
        self.track_to_class = {}
        self.iou_threshold = iou_threshold

    def reset(self):
        """Call when starting a new video/stream."""
        self.tracker = Sort(max_age=self.tracker.max_age)
        self.track_to_class.clear()

    def process(self, boxes, confs, cls_ids):
        dets = self._prepare_sort_input(boxes, confs)
        tracked = self.tracker.update(dets)

        if len(tracked) == 0:
            return self._empty_result()

        self._associate_classes(tracked, boxes, confs, cls_ids)
        return self._format_output(tracked)

    # ---------- Internal helpers ----------

    def _prepare_sort_input(self, boxes, confs):
        if len(boxes) == 0:
            return np.empty((0, 5))
        return np.hstack((boxes, confs.reshape(-1, 1)))

    def _associate_classes(self, tracked, boxes, confs, cls_ids):
        if len(boxes) == 0:
            return

        for i, track_id in enumerate(tracked[:, 8].astype(int)):
            if track_id in self.track_to_class:
                continue

            ious = iou_batch(tracked[i:i + 1, :4], boxes)
            best_idx = np.argmax(ious)
            best_iou = ious[0, best_idx]

            if best_iou > self.iou_threshold:
                self.track_to_class[track_id] = (
                    int(cls_ids[best_idx]),
                    float(confs[best_idx])
                )

    def _format_output(self, tracked):
        out = np.zeros((len(tracked), 10))
        out[:, :9] = tracked

        for i, track_id in enumerate(tracked[:, 8].astype(int)):
            cls, conf = self.track_to_class.get(track_id, (-1, 0.0))
            out[i, 4] = conf
            out[i, 9] = cls

        return {
            "boxes": out[:, :4],
            "conf": out[:, 4],
            "cls": out[:, 9],
            "id": out[:, 8],
        }

    def _empty_result(self):
        return {
            "boxes": np.zeros((0, 4)),
            "conf": [],
            "cls": [],
            "id": [],
        }
