"""
Multi-Stream Face Tracker
Manages face tracking for multiple video streams
"""
from PySide6.QtCore import QObject, Signal
import numpy as np
from typing import Dict, List
from copy import deepcopy

from .multistream_face_recognizer import MultiStreamFaceRecognizer
from .multistream_roi_manager import MultiStreamROIManager
from .authorization_db import AuthorizationDatabase
from .face_tracker import FaceTracker, TrackedFace
from .bytetracker import BYTETracker
import queue
import threading
import time
from .face_recognizer import AnnotatedFrame
from .utils import Framerate


class MultiStreamFaceTracker(QObject):
    """Manages face tracking for multiple streams"""
    frame_ready = Signal(int, np.ndarray)  # (stream_idx, frame)

    def __init__(self, 
                 face_recognizer: MultiStreamFaceRecognizer,
                 auth_database: AuthorizationDatabase,
                 roi_manager: MultiStreamROIManager):
        """
        Initialize multi-stream face tracker
        
        Args:
            face_recognizer: Multi-stream face recognizer instance
            auth_database: Authorization database (shared across streams)
            roi_manager: Multi-stream ROI manager
        """
        super().__init__()
        self.face_recognizer = face_recognizer
        self.auth_database = auth_database
        self.roi_manager = roi_manager
        self.num_streams = face_recognizer.num_streams
        
        # Per-stream trackers
        self.trackers: Dict[int, BYTETracker] = {
            i: BYTETracker() for i in range(self.num_streams)
        }
        
        # Per-stream tracked faces
        self.tracker_dicts: Dict[int, Dict[int, TrackedFace]] = {
            i: {} for i in range(self.num_streams)
        }
        self.tracker_dict_locks: Dict[int, threading.Lock] = {
            i: threading.Lock() for i in range(self.num_streams)
        }
        
        # Per-stream current frames
        self.current_frames: Dict[int, AnnotatedFrame] = {
            i: AnnotatedFrame(np.zeros([1920, 1080, 3])) for i in range(self.num_streams)
        }
        
        # Create worker threads for detection and recognition
        self.detection_thread = MultiStreamDetectionThread(self)
        self.recognition_thread = MultiStreamRecognitionThread(self)

    def start(self):
        """Start detection and recognition threads"""
        self.detection_thread.start()
        self.recognition_thread.start()

    def stop(self):
        """Stop all threads"""
        self.detection_thread.stop()
        self.recognition_thread.stop()
        self.detection_thread.join()
        self.recognition_thread.join()

    def detect(self, stream_idx: int, frame: np.ndarray):
        """Put frame for face detection on specified stream"""
        try:
            self.face_recognizer.detect_put(stream_idx, frame, block=False)
        except queue.Full:
            pass

    def get_activated_tracker_objects(self, stream_idx: int) -> List[TrackedFace]:
        """Get all currently activated tracked faces for a specific stream"""
        with self.tracker_dict_locks[stream_idx]:
            return [deepcopy(obj) for obj in self.tracker_dicts[stream_idx].values() if obj.activated]


class MultiStreamDetectionThread(threading.Thread):
    """Thread that processes face detections for all streams"""
    def __init__(self, multi_tracker: MultiStreamFaceTracker):
        super().__init__(daemon=True)
        self.multi_tracker = multi_tracker
        self.stop_threads = False
        self.refresh_interval = 1
        self.framerates: Dict[int, Framerate] = {
            i: Framerate() for i in range(multi_tracker.num_streams)
        }

    def _update_detections(self, stream_idx: int):
        """Update detections for a specific stream"""
        try:
            annotated_frame = self.multi_tracker.face_recognizer.detect_get(stream_idx, timeout=0.033)
            self.multi_tracker.current_frames[stream_idx] = annotated_frame
            self.framerates[stream_idx].update()
        except queue.Empty:
            return

        # Mark all current tracked objects as not active
        with self.multi_tracker.tracker_dict_locks[stream_idx]:
            for tracked_object in self.multi_tracker.tracker_dicts[stream_idx].values():
                tracked_object.activated = False

            current_time = time.time()
            if annotated_frame.num_detected_faces == 0:
                return

            # Build detections array expected by BYTETracker
            dets = []
            for bbox, score in zip(annotated_frame.boxes, annotated_frame.scores):
                x, y, w, h = bbox
                dets.append(np.array([x, y, x+w, y+h, score, 0]))
            dets = np.array(dets, dtype=np.float32)

            # Update tracker with the new detections
            tracker = self.multi_tracker.trackers[stream_idx]
            roi_manager = self.multi_tracker.roi_manager.get_roi_manager(stream_idx)
            
            for tracklet in tracker.update(dets, None):
                x1, y1, x2, y2, track_id, _, _ = tracklet.astype(int)
                keypoints = self._get_keypoints((x1, y1, x2, y2), annotated_frame)

                # Check which ROI(s) the face is in for this stream
                in_roi_ids = []
                if roi_manager and roi_manager.has_roi():
                    in_roi_ids = roi_manager.in_roi_xyxy((x1, y1, x2, y2))

                # For an existing track, update bbox and activate it
                if track_id in self.multi_tracker.tracker_dicts[stream_idx]:
                    tracked_obj = self.multi_tracker.tracker_dicts[stream_idx][track_id]
                    tracked_obj.bbox = (x1, y1, x2, y2)
                    tracked_obj.keypoints = keypoints
                    tracked_obj.activated = True
                    tracked_obj.in_roi_ids = in_roi_ids
                    tracked_obj.authorized = (tracked_obj.name != "PERSON")

                    # Refresh active track if refresh_interval elapsed
                    if current_time - tracked_obj.last_recognition > self.refresh_interval:
                        try:
                            # Prepare eyes for recognition
                            if len(keypoints) >= 2:
                                eyes = (keypoints[0], keypoints[1])
                            else:
                                eyes = ((x1 + (x2-x1)//4, y1 + (y2-y1)//3), (x1 + 3*(x2-x1)//4, y1 + (y2-y1)//3))
                            self.multi_tracker.face_recognizer.recognize_put(
                                stream_idx, (track_id, annotated_frame.image, (x1, y1, x2, y2), eyes), block=False)
                        except queue.Full:
                            pass
                else:
                    # New track: create a new tracked object and request recognition immediately
                    new_obj = TrackedFace(
                        bbox=(x1, y1, x2, y2), 
                        keypoints=keypoints, 
                        track_id=track_id,
                        name="PERSON",
                        in_roi_ids=in_roi_ids,
                        authorized=False,
                        last_recognition=current_time
                    )
                    self.multi_tracker.tracker_dicts[stream_idx][track_id] = new_obj
                    try:
                        # Prepare eyes for recognition
                        if len(keypoints) >= 2:
                            eyes = (keypoints[0], keypoints[1])
                        else:
                            eyes = ((x1 + (x2-x1)//4, y1 + (y2-y1)//3), (x1 + 3*(x2-x1)//4, y1 + (y2-y1)//3))
                        self.multi_tracker.face_recognizer.recognize_put(
                            stream_idx, (track_id, annotated_frame.image, (x1, y1, x2, y2), eyes), block=False)
                    except queue.Full:
                        pass

    def run(self):
        """Process detections for all streams"""
        while not self.stop_threads:
            for stream_idx in range(self.multi_tracker.num_streams):
                self._update_detections(stream_idx)
                if self.multi_tracker.current_frames[stream_idx]:
                    self.multi_tracker.frame_ready.emit(
                        stream_idx, 
                        self.multi_tracker.current_frames[stream_idx].image
                    )

    def stop(self):
        self.stop_threads = True

    def _get_keypoints(self, track_box, annotated_frame):
        """Re-associate the tracked box with the detected box to extract keypoints"""
        best_iou = 0
        best_idx = None

        # Loop over detections from annotated_frame
        for idx, det_box in enumerate(annotated_frame.boxes):
            # Convert detection box from (x, y, w, h) to (x1, y1, x2, y2)
            det_box_converted = (det_box[0], det_box[1], det_box[0] + det_box[2], det_box[1] + det_box[3])
            iou = self._compute_iou(track_box, det_box_converted)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        return annotated_frame.keypoints[best_idx] if best_idx is not None else []

    def _compute_iou(self, boxA, boxB):
        """Compute IoU between two boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)


class MultiStreamRecognitionThread(threading.Thread):
    """Thread that processes face recognition results for all streams"""
    def __init__(self, multi_tracker: MultiStreamFaceTracker):
        super().__init__(daemon=True)
        self.multi_tracker = multi_tracker
        self.stop_threads = False
        self.framerates: Dict[int, Framerate] = {
            i: Framerate() for i in range(multi_tracker.num_streams)
        }

    def run(self):
        """Process recognition results for all streams"""
        while not self.stop_threads:
            for stream_idx in range(self.multi_tracker.num_streams):
                self._run(stream_idx)
            self.framerates[0].update()  # Update at least one framerate

    def _run(self, stream_idx: int):
        """Process recognition for a specific stream"""
        try:
            # Expect recognition results as (track_id, embedding)
            track_id, embedding = self.multi_tracker.face_recognizer.recognize_get(stream_idx, timeout=0.1)
        except queue.Empty:
            return

        with self.multi_tracker.tracker_dict_locks[stream_idx]:
            if track_id not in self.multi_tracker.tracker_dicts[stream_idx]:
                return

            # Check authorization database
            name, distances = self.multi_tracker.auth_database.find(embedding)
            # If not found, use "PERSON" instead of "Unknown"
            if name == "Unknown":
                name = "PERSON"
                
            tracked_obj = self.multi_tracker.tracker_dicts[stream_idx][track_id]
            tracked_obj.embedding = embedding
            tracked_obj.name = name
            tracked_obj.distances = distances
            tracked_obj.authorized = (name != "PERSON")
            tracked_obj.last_recognition = time.time()
            
            # Update ROI status
            roi_manager = self.multi_tracker.roi_manager.get_roi_manager(stream_idx)
            if roi_manager and roi_manager.has_roi():
                tracked_obj.in_roi_ids = roi_manager.in_roi_xyxy(tracked_obj.bbox)

    def stop(self):
        self.stop_threads = True

