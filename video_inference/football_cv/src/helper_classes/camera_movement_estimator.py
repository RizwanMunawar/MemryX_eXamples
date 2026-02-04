from queue import Queue
from PyQt5.QtCore import QThread
import cv2
import numpy as np

class CameraMovementEstimator():
    def __init__(self, frame, scale_factor=0.5):
        self.scale_factor = scale_factor
        self.inv_scale = 1.0 / scale_factor

        h, w = frame.shape[:2]
        self.new_h = int(h * scale_factor)
        self.new_w = int(w * scale_factor)
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=1, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.03)
        )
        
        first_frame_small = cv2.resize(frame, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        self.old_gray = cv2.cvtColor(first_frame_small, cv2.COLOR_BGR2GRAY)
        
        mask_features = np.zeros_like(self.old_gray)
        top_strip_h = int(self.new_h * 0.12)
        bottom_strip_start_h = int(self.new_h * 0.85)
        mask_features[0:top_strip_h, :] = 1 
        mask_features[bottom_strip_start_h:self.new_h, :] = 1

        self.features_params = dict(
            maxCorners=50, 
            qualityLevel=0.3,
            minDistance=3, 
            blockSize=7,
            mask=mask_features
        )
        
        self.old_features = cv2.goodFeaturesToTrack(self.old_gray, **self.features_params)
        
        self.minimum_features = 25
        self.is_first_frame = True 
        
        self.smoothed_movement = np.array([0.0, 0.0])
        self.alpha = 0.25
        
        self.cumulative_x = 0.0
        self.cumulative_y = 0.0
        self.latest_x = 0
        self.latest_y = 0

    def get_camera_movement_realtime(self, new_frame):
        if self.is_first_frame:
            self.is_first_frame = False
            return (0, 0)

        frame_small = cv2.resize(new_frame, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        if self.old_features is None or len(self.old_features) < 5:
            self.old_features = cv2.goodFeaturesToTrack(self.old_gray, **self.features_params)
            if self.old_features is None:
                self.old_gray = frame_gray
                return tuple(self.smoothed_movement.tolist())

        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.old_features, None, **self.lk_params
        )

        matched = (status == 1).ravel()
        good_new = new_features[matched]
        good_old = self.old_features[matched]

        if len(good_new) > 0:
            displacement = np.median(good_new - good_old, axis=0).ravel()
            current_movement = displacement * self.inv_scale
        else:
            current_movement = np.array([0.0, 0.0])
        
        self.smoothed_movement = (self.alpha * current_movement) + ((1 - self.alpha) * self.smoothed_movement)
        smoothed_x, smoothed_y = self.smoothed_movement.tolist()

        if len(good_new) < self.minimum_features:
            new_detected = cv2.goodFeaturesToTrack(frame_gray, **self.features_params)
            self.old_features = new_detected if new_detected is not None else good_new
        else:
            self.old_features = good_new.reshape(-1, 1, 2)
        
        self.old_gray = frame_gray
        self.cumulative_x += smoothed_x
        self.cumulative_y += smoothed_y
        self.latest_x = smoothed_x
        self.latest_y = smoothed_y

        return (smoothed_x, smoothed_y)
    
    def get_all(self):
        return self.latest_x, self.latest_y, self.cumulative_x, self.cumulative_y

# ================================
# Optimized Camera Worker
# ================================
class CameraWorker(QThread):
    def __init__(self):
        super().__init__()
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.active = True

    def add_frame(self, frame):
        self.input_queue.put(frame)

    def get_result(self):
        return self.output_queue.get()

    def run(self):
        # Initialization (Happens once)
        # Block and wait for the very first frame
        first_frame = self.input_queue.get()
        
        # Safety check: If stop() was called before any frame arrived
        if first_frame is None:
            return

        # Initialize the estimator using the first frame
        self.estimator = CameraMovementEstimator(first_frame)
        
        # Process the first frame immediately
        self.estimator.get_camera_movement_realtime(first_frame)
        self.output_queue.put(self.estimator.get_all())

        # The "Hot" Loop (Zero overhead)
        # We know estimator exists, so we just run.
        while self.active:
            frame = self.input_queue.get()
            
            if frame is None:
                break
            
            self.estimator.get_camera_movement_realtime(frame)
            self.output_queue.put(self.estimator.get_all())

    def stop(self):
        self.active = False
        self.input_queue.put(None)
        self.wait()