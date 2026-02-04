from typing import List, Tuple
import cv2
import numpy as np
import onnx 
from memryx import AsyncAccl
import onnxruntime as ort
from dataclasses import dataclass, field
from queue import Queue, Empty
import os
import time

from mx_det import Detect
from mx_rtmpose import Rtmpose
from bytetracker import BYTETracker
import threading

@dataclass 
class Pose():
    track_id: int
    bbox: np.ndarray       = field(default_factory=lambda: [])
    center: float          = 0.0
    scale: float           = 0.0
    kpots: np.ndarray      = field(default_factory=lambda: [])
    score: float           = 0.0
    name: int = -1


@dataclass
class AnnotatedFrame():
    image: np.ndarray
    num_detections: int = 0
    poses: List[Pose] = field(default_factory=lambda: [])

@dataclass
class TrackedObject:
    bbox: Tuple[int, int, int, int]
    track_id: int = -1
    name: int = -1
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros([1]))


class Mxpose:
    def __init__(self, mx_modeldir,**kwargs):
        self._stopped            = False 
        self._outstanding_frames = 0

        self.tracker = BYTETracker(
            track_thresh=0.5,      # High confidence threshold
            match_thresh=0.7,      # IoU threshold for matching
            track_buffer=30,       # Keep track for 30 frames
            frame_rate=30)
        self.tracker_dict = {}  # Mapping from track_id to TrackedObject
        self.tracker_num = 0
        
        self.det_model = Detect()
        self.rtmpose_model = Rtmpose()

        self.input_q = Queue()  # (annotated_frame)
        self.stage0_q = Queue()  # (annotated_frame, ratio)
        self.stage1_q = Queue()  # (annotated_frame + box)
        self.stage2_q = Queue()  # (annotated_frame, box, pose info)
        self.output_q = Queue()  # (annotated_frame with results)

        self.dfp_path = os.path.join(mx_modeldir, 'models/models.dfp')
        self.det_postprocess = os.path.join(mx_modeldir, 'models/model_0_yolox_post.onnx')
        self.det_preprocess = os.path.join(mx_modeldir, 'models/model_0_yolox_pre.onnx')

        self.accl = AsyncAccl(self.dfp_path,device_ids=0)
        self.accl.set_preprocessing_model(self.det_preprocess, model_idx=0)
        self.accl.set_postprocessing_model(self.det_postprocess, model_idx=0)

        self.accl.connect_input(self.detect_source, model_idx=0)
        self.accl.connect_output(self.detect_sink, model_idx=0)
        self.accl.connect_input(self.pose_source, model_idx=1)
        self.accl.connect_output(self.pose_sink, model_idx=1)

    def put(self, image, block=True, timeout=None):
        annotated_frame = AnnotatedFrame(np.array(image))
        self.input_q.put(annotated_frame, block, timeout)      
        self._outstanding_frames += 1

    def get(self, block=True, timeout=None):
        self._outstanding_frames -= 1
        annotated_frame = self.output_q.get(block, timeout)
        self.output_q.task_done()
        return annotated_frame
    
    def __del__(self):
        if not self._stopped:
            self.stop()

    def stop(self):
        while self._outstanding_frames > 0:
            try:
                self.get(timeout=0.1)
            except Empty:
                continue
            
        self.input_q.put(None)
        self.stage1_q.put(None)
        self._stopped = True

    def empty(self):
        return self.output_q.empty() and self.input_q.empty()  

    def full(self):
        return self.input_q.full()
    

    def detect_source(self):
        annotated_frame = self.input_q.get()
        if annotated_frame is None:
            return None

        img = annotated_frame.image.copy()
        padded_img, ratio = self.det_model.preprocess(img)
        self.stage0_q.put((annotated_frame, ratio))
        
        img = padded_img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img[None, :, :, :].astype(np.float32)

    def detect_sink(self, *outputs):
        annotated_frame, ratio = self.stage0_q.get()
        final_boxes = self.det_model.postprocess(outputs[0], ratio)

        annotated_frame.num_detections = len(final_boxes)
        if annotated_frame.num_detections == 0:
            self.output_q.put((annotated_frame))
            return
        
        dets = []
        for bbox in final_boxes:
            x1, y1, x2, y2 = bbox
            dets.append(np.array([x1, y1, x2, y2, 1, 1]))
        dets = np.array(dets, dtype=np.float32)

        person_tracks = self.tracker.update(dets, None)

        annotated_frame.num_detections = len(person_tracks)
        for tracklet in person_tracks:
            x1, y1, x2, y2, track_id, score, _ = tracklet.astype(int)
            if track_id in self.tracker_dict:
                tracked_obj = self.tracker_dict[track_id]
                tracked_obj.bbox = (x1, y1, x2, y2)
            else:
                new_obj = TrackedObject(
                    bbox=(x1, y1, x2, y2), 
                    track_id=track_id, 
                    name = "tracked " + str(track_id)
                )
                self.tracker_dict[track_id] = new_obj
            self.stage1_q.put((track_id,annotated_frame, (x1,y1,x2,y2)),block=False)


    def pose_source(self):
        data = self.stage1_q.get()
        if data is None:
            return None 
        track_id,annotated_frame, bbox = data

        resized_img, center, scale = self.rtmpose_model.preprocess(annotated_frame.image.copy(), bbox)
        self.stage2_q.put((track_id, annotated_frame, bbox,center, scale))

        resized_img = resized_img.transpose(2,0,1)
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)
        return resized_img[None, :, :, :].astype(np.float32)

    def pose_sink(self, *outputs):
        track_id, annotated_frame, bbox,center, scale = self.stage2_q.get()
        keypoints, scores = self.rtmpose_model.postprocess(outputs, center, scale)

        if keypoints is None:
            self.output_q.put((annotated_frame))
            return
        annotated_frame.poses.append(Pose(track_id,bbox, center, scale, keypoints, scores, self.tracker_dict[track_id].name))

        if len(annotated_frame.poses) == annotated_frame.num_detections:
            self.output_q.put((annotated_frame))







            

