"""
Face Tracker with ROI support
Adapted from realtime_multiface_recognition/src/modules/tracker.py
"""
from PySide6.QtCore import QThread, Signal, QObject
from .bytetracker import BYTETracker
import queue
from dataclasses import dataclass, field
import numpy as np
from .face_recognizer import FaceRecognizer, AnnotatedFrame
from pathlib import Path
import time
from .authorization_db import AuthorizationDatabase
from .utils import Framerate
import threading
from copy import deepcopy


@dataclass
class TrackedFace:
    """Represents a tracked face with ROI and authorization status"""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    keypoints: list[tuple[int, int]]
    track_id: int
    name: str = "PERSON"  # Changed from "Unknown" to "PERSON"
    activated: bool = True
    last_recognition: float = 0.0
    distances: list = field(default_factory=list)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros([128]))
    in_roi_ids: list = field(default_factory=list)  # List of ROI IDs this face is in
    authorized: bool = False  # True if name != "PERSON"
    
    # Backward compatibility property
    @property
    def in_roi(self) -> bool:
        """Check if face is in any ROI (backward compatibility)"""
        return len(self.in_roi_ids) > 0


def compute_iou(boxA, boxB):
    # boxA and boxB are (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


class FaceTracker(QObject):
    """Face tracker with ROI and authorization support"""
    frame_ready = Signal(np.ndarray)

    def __init__(self, face_recognizer: FaceRecognizer, auth_database: AuthorizationDatabase, roi_manager=None):
        super().__init__()
        self.tracker = BYTETracker()
        self.face_recognizer = face_recognizer
        self.auth_database = auth_database
        self.roi_manager = roi_manager
        self.tracker_dict = {}  # Mapping from track_id to TrackedFace
        self.tracker_dict_lock = threading.Lock()
        self.current_frame = AnnotatedFrame(np.zeros([1920, 1080, 3]))
        self.composite_queue = queue.Queue(maxsize=1)
        
        # Create worker threads for detection and recognition
        self.detection_thread = DetectionThread(self)
        self.recognition_thread = RecognitionThread(self)

    def start(self):
        self.detection_thread.start()
        self.recognition_thread.start()

    def stop(self):
        self.detection_thread.stop()
        self.recognition_thread.stop()
        self.detection_thread.wait()
        self.recognition_thread.wait()
        if hasattr(self.face_recognizer, 'stop'):
            self.face_recognizer.stop()

    def detect(self, frame):
        """Put frame for face detection"""
        try:
            self.face_recognizer.detect_put(frame, block=False)
        except queue.Full:
            pass

    def get_activated_tracker_objects(self) -> list:
        """Get all currently activated tracked faces"""
        with self.tracker_dict_lock:
            return [deepcopy(obj) for obj in self.tracker_dict.values() if obj.activated]


class DetectionThread(QThread):
    """Thread that processes face detections and updates tracker"""
    def __init__(self, face_tracker):
        super().__init__()
        self.face_tracker = face_tracker
        self.stop_threads = False
        self.refresh_interval = 1
        self.framerate = Framerate()

    def _update_detections(self):
        try:
            annotated_frame = self.face_tracker.face_recognizer.detect_get(timeout=0.033)
            self.face_tracker.current_frame = annotated_frame
            self.framerate.update()
        except queue.Empty:
            return

        # Mark all current tracked objects as not active
        with self.face_tracker.tracker_dict_lock:
            for tracked_object in self.face_tracker.tracker_dict.values():
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
            for tracklet in self.face_tracker.tracker.update(dets, None):
                x1, y1, x2, y2, track_id, _, _ = tracklet.astype(int)
                keypoints = self._get_keypoints((x1, y1, x2, y2), annotated_frame)

                # Check which ROI(s) the face is in
                in_roi_ids = []
                if self.face_tracker.roi_manager and self.face_tracker.roi_manager.has_roi():
                    in_roi_ids = self.face_tracker.roi_manager.in_roi_xyxy((x1, y1, x2, y2))

                # For an existing track, update bbox and activate it
                if track_id in self.face_tracker.tracker_dict:
                    tracked_obj = self.face_tracker.tracker_dict[track_id]
                    tracked_obj.bbox = (x1, y1, x2, y2)
                    tracked_obj.keypoints = keypoints
                    tracked_obj.activated = True
                    tracked_obj.in_roi_ids = in_roi_ids
                    tracked_obj.authorized = (tracked_obj.name != "PERSON")

                    # Refresh active track if refresh_interval elapsed
                    if current_time - tracked_obj.last_recognition > self.refresh_interval:
                        try:
                            # Prepare eyes for recognition (use keypoints if available, otherwise estimate)
                            if len(keypoints) >= 2:
                                eyes = (keypoints[0], keypoints[1])
                            else:
                                # Estimate eye positions from bbox
                                eyes = ((x1 + (x2-x1)//4, y1 + (y2-y1)//3), (x1 + 3*(x2-x1)//4, y1 + (y2-y1)//3))
                            self.face_tracker.face_recognizer.recognize_put(
                                (track_id, annotated_frame.image, (x1, y1, x2, y2), eyes), block=False)
                        except queue.Full:
                            pass
                else:
                    # New track: create a new tracked object and request recognition immediately
                    new_obj = TrackedFace(
                        bbox=(x1, y1, x2, y2), 
                        keypoints=keypoints, 
                        track_id=track_id,
                        name="PERSON",  # Default to "PERSON" instead of "Unknown"
                        in_roi_ids=in_roi_ids,
                        authorized=False,
                        last_recognition=current_time
                    )
                    self.face_tracker.tracker_dict[track_id] = new_obj
                    try:
                        # Prepare eyes for recognition (use keypoints if available, otherwise estimate)
                        if len(keypoints) >= 2:
                            eyes = (keypoints[0], keypoints[1])
                        else:
                            # Estimate eye positions from bbox
                            eyes = ((x1 + (x2-x1)//4, y1 + (y2-y1)//3), (x1 + 3*(x2-x1)//4, y1 + (y2-y1)//3))
                        self.face_tracker.face_recognizer.recognize_put(
                            (track_id, annotated_frame.image, (x1, y1, x2, y2), eyes), block=False)
                    except queue.Full:
                        pass

    def run(self):
        while not self.stop_threads:
            self._update_detections()
            if self.face_tracker.current_frame:
                self.face_tracker.frame_ready.emit(self.face_tracker.current_frame.image)

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
            iou = compute_iou(track_box, det_box_converted)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        return annotated_frame.keypoints[best_idx] if best_idx is not None else []


class RecognitionThread(QThread):
    """Thread that processes face recognition results"""
    def __init__(self, face_tracker):
        super().__init__()
        self.face_tracker = face_tracker
        self.stop_threads = False
        self.framerate = Framerate()

    def run(self):
        while not self.stop_threads:
            self.framerate.update()
            self._run()

    def _run(self):
        try:
            # Expect recognition results as (track_id, embedding)
            track_id, embedding = self.face_tracker.face_recognizer.recognize_get(timeout=0.1)
        except queue.Empty:
            return

        with self.face_tracker.tracker_dict_lock:
            if track_id not in self.face_tracker.tracker_dict:
                return

            # Check authorization database
            name, distances = self.face_tracker.auth_database.find(embedding)
            # If not found, use "PERSON" instead of "Unknown"
            if name == "Unknown":
                name = "PERSON"
                
            tracked_obj = self.face_tracker.tracker_dict[track_id]
            tracked_obj.embedding = embedding
            tracked_obj.name = name
            tracked_obj.distances = distances
            tracked_obj.authorized = (name != "PERSON")
            tracked_obj.last_recognition = time.time()
            
            # Update ROI status
            if self.face_tracker.roi_manager and self.face_tracker.roi_manager.has_roi():
                tracked_obj.in_roi_ids = self.face_tracker.roi_manager.in_roi_xyxy(tracked_obj.bbox)

    def stop(self):
        self.stop_threads = True

