# OpenCV and helper libraries imports
import sys
import os
from queue import Queue
import cv2 as cv
import numpy as np
from memryx import AsyncAccl
from typing import List, Tuple, Optional
import argparse
import time
from dataclasses import dataclass, field
from collections import deque
from dtaidistance import dtw_ndim, dtw

# Config / Defines
@dataclass
class Config:
    # DTW / scoring parameters
    SCORE_THRESHOLD: float = 0.5    # Min score to be accepted (0~1). Suggest: 0.50~0.80
    DTW_ALPHA: float = 15.0         # exp(-alpha * dist_norm). Suggest: 6~20
    DTW_WINDOW_EXTRA: int = 30      # DTW window = abs(Tu-Tr) + extra. Suggest: 10~90

    # Real-time segmentation parameters
    EVAL_WINDOW: int = 45           # Smoothing window (frames). Suggest: 30~60
    EVAL_INTERVAL: int = 5          # Evaluation interval (frames). Suggest: 3~10
    PERFECT_SCORE: float = 0.90     # Perfect rep score (for UI). Suggest: 0.80~0.9

    # Video clip setting
    CACHE_VIDEO_W: int = 640                # Cached coach video width. Suggested: 320~1280
    CACHE_VIDEO_H: int = 360                # Cached coach video height. Suggested: 180~720
    FIXED_DISPLAY_HEIGHT: int = 540         # Display height for both panels. Suggested: 360~720 (depends on monitor/GPU)
    PREFIX: str = "coach"
    FILENAME_VID: str = field(init=False)   # Saved processed video filename
    FILENAME_NPY: str = field(init=False)   # Saved processed keypoints filename
    def __post_init__(self):
        self.FILENAME_VID = f"{self.PREFIX}_action.mp4"
        self.FILENAME_NPY = f"{self.PREFIX}_action.npy"

# ALGORITHM: Dynamic Time Warping & Similarity Calculation
class ActionComparator:
    def __init__(self, reference_kpts: np.ndarray, cfg: Config):
        self.cfg = cfg
        self.ref_centered = self._center_pose(reference_kpts)
        self.ref_data = self.ref_centered[:, :, :2].reshape(reference_kpts.shape[0], -1)

    # Pose Normalization & Centering
    # To compare actions effectively, we must remove translation and scale differences.
    # 1. Centering: Shifts the skeleton so the root (mid-hip) is at (0,0).
    # 2. Normalization: Scales the limb vectors so the action is independent of user height/distance.
    def _center_pose(self, kpts_seq):
        centered_seq = kpts_seq.copy()
        for i in range(len(centered_seq)):
            hip_left = centered_seq[i, 11, :2]
            hip_right = centered_seq[i, 12, :2]
            root = (hip_left + hip_right) / 2
            centered_seq[i, :, 0] -= root[0]
            centered_seq[i, :, 1] -= root[1]
            shoulder_left = centered_seq[i, 5, :2]
            shoulder_right = centered_seq[i, 6, :2]
            shoulder_mid = (shoulder_left + shoulder_right) / 2
            torso_len = np.linalg.norm(shoulder_mid)
            if torso_len < 1e-6:
                torso_len = 1.0

            centered_seq[i, :, :2] /= torso_len

        return centered_seq

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        return data / norms

    def compare_pair(self, ref_kpts_seq: np.ndarray, user_kpts_seq: np.ndarray) -> float:
        if dtw_ndim is None:
            return 0.0, 0.0, float("inf"), 0.0, 0

        dist = float("inf")
        path_len = 0
        ref_centered = self._center_pose(ref_kpts_seq)
        ref_xy = ref_centered[:, :, :2].reshape(ref_kpts_seq.shape[0], -1)
        ref_norm = self._normalize(ref_xy)

        user_centered = self._center_pose(user_kpts_seq)
        conf = user_centered[:, :, 2]
        mask = (conf > 0).astype(np.float32)
        user_xy = user_centered[:, :, :2].copy()
        user_xy[mask == 0] = 0.0
        user_feat = user_xy.reshape(user_kpts_seq.shape[0], -1)
        user_norm = self._normalize(user_feat)

        ref = np.nan_to_num(np.asarray(ref_norm, dtype=np.double), nan=0.0, posinf=0.0, neginf=0.0)
        usr = np.nan_to_num(np.asarray(user_norm, dtype=np.double), nan=0.0, posinf=0.0, neginf=0.0)

        base = abs(usr.shape[0] - ref.shape[0])
        window = base + int(self.cfg.DTW_WINDOW_EXTRA)

        # Dynamic Time Warping (DTW)
        # Standard Euclidean distance fails if the user is faster/slower than the reference.
        # DTW finds the optimal alignment path between two temporal sequences (User vs Ref),
        # allowing for non-linear stretching or compressing of time to measure similarity.
        try:
            dist, mat = dtw_ndim.warping_paths_fast(ref, usr, window=window)
            path = dtw.best_path(mat)
            path_len = len(path)
        except Exception as e:
            dist, mat = dtw_ndim.warping_paths_fast(ref, usr, window=None)
            path = dtw.best_path(mat)
            path_len = len(path)

        if (not np.isfinite(dist)) or path_len <= 0:
            return 0.0, 0.0, dist, 0.0, path_len

        if (not np.isfinite(dist)) or path_len <= 0:
            return 0.0, 0.0, dist, 0.0, path_len

        dist_norm = float(dist) / float(path_len)
        alpha = float(self.cfg.DTW_ALPHA)
        score_raw = float(np.exp(-alpha * dist_norm))
        thr = float(self.cfg.SCORE_THRESHOLD)
        score_final = score_raw if score_raw >= thr else 0.0
        return score_raw, score_final, float(dist), float(dist_norm), int(path_len)

# Main Application Class
class App:
    def __init__(self, cam_source, model_input_shape, mirror=False, src_is_cam=True, cfg: Config = Config(), input_video_path: Optional[str] = None, **kwargs):
        self.cfg = cfg
        self.cache_w = self.cfg.CACHE_VIDEO_W
        self.cache_h = self.cfg.CACHE_VIDEO_H
        self.data_path = self.cfg.FILENAME_NPY
        self.video_path = self.cfg.FILENAME_VID
        self.ref_kpts_list = []
        self.ref_frames_cache = []
        self.live_mode_initialized = False
        self.display_canvas = None
        self.target_h = 0
        self.target_w = 0
        self.ref_w_final = 0
        self.fixed_display_height = self.cfg.FIXED_DISPLAY_HEIGHT
        self.cam_source = cam_source
        self.cam = None
        self.model_input_shape = model_input_shape
        self.ratio = None
        self.mirror = mirror                    # Mirror user view (webcam). Suggested: True for front camera UX
        self.capture_queue = Queue(maxsize=5)   # Frame queue depth. Suggested: 2~10 (bigger = more latency, smoother)
        self.box_score = 0.25                   # YOLO detection confidence threshold. Suggested: 0.10~0.50 (lower = more boxes, noisier). :contentReference[oaicite:2]{index=2}
        self.kpt_score = 0.5                    # Keypoint confidence threshold. Suggested: 0.30~0.70 (higher = fewer but cleaner kpts)
        self.nms_thr = 0.2                      # NMS IoU threshold. Suggested: 0.2~0.6 (lower = suppress more boxes). :contentReference[oaicite:3]{index=3}
        self.src_is_cam = src_is_cam

        # Predefined color list for drawing keypoints
        self.COLOR_LIST = list([[128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
                            [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
                            [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0],
                            [255, 0, 51], [153, 0, 153], [51, 0, 51], [0, 0, 0],
                            [0, 102, 255], [0, 51, 255], [0, 153, 255], [0, 153, 153]])

        # Define keypoint pairs for drawing skeletons
        self.KEYPOINT_PAIRS = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        self.comparator = None
        self.frame_counter = 0
        self.current_score = 0.0
        self.rep_count = 0          # total completed reps in LIVE
        self.miss_det_cnt = 0       # consecutive missing detections
        self.playback_idx = 0
        self._fps_last_t = time.perf_counter()
        self._fps_frames = 0
        self._fps_value = 0.0
        self._pp_ms_acc = 0.0
        self._pp_cnt = 0
        self._post_ms_acc = 0.0
        self._post_cnt = 0
        self._dtw_ms_acc = 0.0
        self._dtw_cnt = 0
        self._print_every = 30      # Print [PERF] every N postprocess calls. Suggested: 15~120
        self._preprocess_t0 = None
        self.user_buffer = deque(maxlen=self.cfg.EVAL_WINDOW)
        self.ui_status_text = "Ready"
        self.ui_status_color = (255, 255, 255)
        self.rep_trigger_active = False
        self.video_source = input_video_path  # Used in ANALYZING mode
        self.live_cap = None
        self.prev_center = None
        self.prev_torso_len = 0

        if isinstance(self.cam_source, str) and os.path.exists(self.cam_source):
             print(f"Pre-calculating size from: {self.cam_source}")
             tmp_cap = cv.VideoCapture(self.cam_source)
             if tmp_cap.isOpened():
                 ret, tmp_frame = tmp_cap.read()
                 if ret:
                     real_h, real_w = tmp_frame.shape[:2]
                     scale_factor = self.fixed_display_height / real_h
                     self.target_w = int(real_w * scale_factor)
                     self.target_h = self.fixed_display_height
                     print(f"Size Locked: {self.target_w}x{self.target_h}")
                 tmp_cap.release()

        if self.load_reference_data():
            print(">>> Found local reference files, enter LIVE mode")
            self.state = "LIVE"
            self.live_cap = None
            self.switch_to_live_mode()
        else:
            self.state = "ANALYZING"

            if not self.video_source:
                print("Error: --input_video is required when reference cache is missing.")
                sys.exit(1)
            if not os.path.exists(self.video_source):
                print(f"Error: Input video not found: {self.video_source}")
                sys.exit(1)

            self.live_cap = cv.VideoCapture(self.video_source, cv.CAP_FFMPEG)
            if not self.live_cap.isOpened():
                print(f"Error: Could not open input video: {self.video_source}")
                sys.exit(1)

            # Probe first frame to fail fast on decode issues
            ok, probe = self.live_cap.read()
            if not ok or probe is None:
                print(f"Error: Could not decode first frame from: {self.video_source}")
                self.live_cap.release()
                sys.exit(1)
            self.live_cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    def load_reference_data(self):
        if not os.path.exists(self.data_path) or not os.path.exists(self.video_path):
            return False

        try:
            self.ref_kpts_list = list(np.load(self.data_path))
            cap = cv.VideoCapture(self.video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.ref_frames_cache.append(frame)
            cap.release()

            if len(self.ref_kpts_list) > 0:
                ref_data = np.array(self.ref_kpts_list)
                self.comparator = ActionComparator(ref_data, self.cfg)

            # Basic consistency check
            if len(self.ref_kpts_list) == 0 or len(self.ref_frames_cache) == 0:
                return False
            return True
        except Exception as e:
            print(f"Load saved data failed: {e}")
            self.ref_kpts_list = []
            self.ref_frames_cache = []
            return False

    def save_reference_data(self):
        try:
            np.save(self.data_path, np.array(self.ref_kpts_list))
            if len(self.ref_frames_cache) > 0:
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                out = cv.VideoWriter(self.video_path, fourcc, 30.0, (self.cache_w, self.cache_h))

                for frame in self.ref_frames_cache:
                    out.write(frame)
                out.release()

            print(f"Save success, Data: {self.data_path}, video: {self.video_path}")
        except Exception as e:
            print(f"Error, save failed: {e}")

    def generate_frame(self):
        while True:
            frame = None
            if self.state == "ANALYZING":
                ok, frame = self.live_cap.read()
                if not ok or frame is None:
                    print(f"Analysis Complete! Frames: {len(self.ref_kpts_list)}. Switching to Live Mode...")
                    self.switch_to_live_mode()
                    continue

                frame = cv.resize(frame, (self.cache_w, self.cache_h))

            elif self.state == "LIVE":
                if self.cam is None:
                    return None
                ok, frame = self.cam.read()
                if not ok or frame is None:
                    if isinstance(self.cam_source, str) and os.path.exists(self.cam_source):
                        self.cam.set(cv.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = self.cam.read()
                        if not ok or frame is None:
                            print("Error: Could not read from user video after resetting.")
                            return None
                    else:
                        print("EOF Camera")
                        return None

                if isinstance(self.cam_source, str):
                    time.sleep(0.033)
                if self.mirror:
                    frame = cv.flip(frame, 1)
                if self.target_h > 0 and self.target_w > 0:
                    frame = cv.resize(frame, (self.target_w, self.target_h))

            if self.capture_queue.full():
                if self.state == "LIVE":
                    continue

            self.capture_queue.put(frame)
            self._preprocess_t0 = time.perf_counter()
            out, self.ratio = self.preprocess_image(frame)
            t1 = time.perf_counter()
            self._pp_ms_acc += (t1 - self._preprocess_t0) * 1000.0
            self._pp_cnt += 1
            return out

    def switch_to_live_mode(self):
        self.state = "LIVE"
        if self.live_cap is not None:
            self.live_cap.release()
            self.live_cap = None
            print(">>> Reference video source released.")
        if len(self.ref_kpts_list) > 0 and not os.path.exists(self.data_path):
            self.save_reference_data()

        if len(self.ref_kpts_list) > 0:
            ref_data = np.array(self.ref_kpts_list)
            self.comparator = ActionComparator(ref_data, self.cfg)

        if isinstance(self.cam_source, str):
            if not os.path.exists(self.cam_source):
                print(f"Error: Video file not found at: {self.cam_source}")
                sys.exit(1)
            print(f"Opening User Video: {self.cam_source}")
        else:
            print(f"Opening Camera Index: {self.cam_source}")

        if self.cam is None:
            temp_cap = cv.VideoCapture(self.cam_source)
            if not temp_cap.isOpened():
                print(f"Error: Could not open source: {self.cam_source}")
                sys.exit(1)

            ret, tmp_frame = temp_cap.read()
            if ret and tmp_frame is not None:
                real_h, real_w = tmp_frame.shape[:2]
                self.target_h = self.fixed_display_height
                scale_factor = self.fixed_display_height / real_h
                self.target_w = int(real_w * scale_factor)
                print(f"Source Raw: {real_w}x{real_h} -> Resized Display: {self.target_w}x{self.target_h}")
                if isinstance(self.cam_source, str):
                    temp_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            else:
                print("Error: Opened source but failed to read first frame.")
                temp_cap.release()
                sys.exit(1)

            temp_cap.release()

            self.cam = cv.VideoCapture(self.cam_source)
            if not self.cam.isOpened():
                print("Error: Could not open camera/source.")
                sys.exit(1)

            if isinstance(self.cam_source, str):
                self.cam.set(cv.CAP_PROP_POS_FRAMES, 0)

            if len(self.ref_frames_cache) > 0:
                print("Pre-resizing...")
                new_cache = []
                scale = self.target_h / self.cache_h
                new_ref_w = int(self.cache_w * scale)
                scale_x = new_ref_w / self.cache_w
                scale_y = self.target_h / self.cache_h

                for frame in self.ref_frames_cache:
                    resized = cv.resize(frame, (new_ref_w, self.target_h), interpolation=cv.INTER_LINEAR)
                    new_cache.append(resized)
                self.ref_frames_cache = new_cache

                for i in range(len(self.ref_kpts_list)):
                    pose = self.ref_kpts_list[i].copy()
                    pose[:, 0] *= scale_x
                    pose[:, 1] *= scale_y
                    self.ref_kpts_list[i] = pose

                print("Pre-resizing done.")

                ref_w_final = self.ref_frames_cache[0].shape[1]
                total_w = ref_w_final + self.target_w
                self.display_canvas = np.zeros((self.target_h, total_w, 3), dtype=np.uint8)
                self.ref_w_final = ref_w_final

                self.live_mode_initialized = True
            else:
                print("Error: Camera opened but failed to read first frame.")
                sys.exit(1)

    # Letterbox Resizing
    # Resizes the image to the model input shape (e.g., 640x640) while maintaining the
    # original aspect ratio. Padding is added to the remaining areas.
    # This prevents the human body from appearing stretched or squashed, which would degrade pose accuracy.
    def preprocess_image(self, image):
        h, w = image.shape[:2]
        r = min(self.model_input_shape[0] / h, self.model_input_shape[1] / w)
        image_resized = cv.resize(image, (int(w * r), int(h * r)), interpolation=cv.INTER_LINEAR)
        frame_rgb = cv.cvtColor(image_resized, cv.COLOR_BGR2RGB)

        padded_img = np.ones((self.model_input_shape[0], self.model_input_shape[1], 3), dtype=np.uint8) * 114
        padded_img[:int(h * r), :int(w * r)] = frame_rgb

        padded_img = padded_img / 255.0
        padded_img = padded_img.astype(np.float32)

        padded_img = np.transpose(padded_img, (2, 0, 1))  # Change shape to (3, 640, 640)
        padded_img = np.expand_dims(padded_img, axis=0)   # Add batch dimension to make it (1, 3, 640, 640)

        return padded_img, r

    def xywh2xyxy(self, box: np.ndarray) -> np.ndarray:
        # Convert bounding boxes from [x, y, w, h] format to [x1, y1, x2, y2] format
        box_xyxy = box.copy()
        box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
        box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
        box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
        box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
        return box_xyxy

    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        '''
        box and boxes are in format [x1, y1, x2, y2]
        '''
        # Calculate intersection area
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Calculate union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        return inter_area / union_area  # Return IoU

    # Non-Maximum Suppression (NMS)
    # The object detector may output multiple overlapping boxes for the same person.
    # NMS iterates through sorted detections, keeping the most confident one and
    # suppressing others that have a high Intersection-over-Union (IoU) overlap.
    def nms_process(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        # Apply non-maximum suppression to reduce redundant overlapping boxes
        sorted_idx = np.argsort(scores)[::-1]  # Sort scores in descending order
        keep_idx = []
        while sorted_idx.size > 0:
            idx = sorted_idx[0]  # Keep the box with the highest score
            keep_idx.append(idx)
            ious = self.compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])  # Calculate IoU
            rest_idx = np.where(ious < iou_thr)[0]  # Keep boxes with IoU below threshold
            sorted_idx = sorted_idx[rest_idx + 1]
        return keep_idx

    def draw_skeleton(self, img, kpts, scores=None):
        for pair in self.KEYPOINT_PAIRS:
            idx1, idx2 = pair
            if idx1 >= len(kpts) or idx2 >= len(kpts):
                continue

            p1 = kpts[idx1]
            p2 = kpts[idx2]

            if scores is not None:
                if scores[idx1] < self.kpt_score or scores[idx2] < self.kpt_score:
                    continue
            elif p1[2] == -1 or p2[2] == -1:
                continue

            cv.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255), 2)

        for i, p in enumerate(kpts):
            if scores is not None and scores[i] < self.kpt_score:
                continue
            if scores is None and p[2] == -1:
                continue
            cv.circle(img, (int(p[0]), int(p[1])), 4, self.COLOR_LIST[i % 5], -1)

    def _count_valid_kpts(self, pose17x3: np.ndarray) -> int:
        conf = pose17x3[:, 2]
        return int(np.sum(conf > self.kpt_score))

    def _pick_user_idx(self, final_boxes, final_kpts) -> Optional[int]:
        if final_kpts is None or len(final_kpts) == 0:
            self.prev_center = None
            self.prev_torso_len = 0
            return None

        best_i = None
        # Strategy A: Tracking (Distance + Size Check)
        if self.prev_center is not None and self.prev_torso_len > 0:
            min_dist = 1e9
            best_tracker_i = None

            for i in range(len(final_boxes)):
                x1, y1, x2, y2 = final_boxes[i]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dist = np.linalg.norm(np.array([cx, cy]) - self.prev_center)
                pose = final_kpts[i].reshape(17, 3)
                shoulder_mid = (pose[5, :2] + pose[6, :2]) / 2
                hip_mid = (pose[11, :2] + pose[12, :2]) / 2
                box_h = y2 - y1
                size_diff = abs(box_h - self.prev_torso_len) / self.prev_torso_len
                if dist < 400 and size_diff < 0.6:
                    if dist < min_dist:
                        min_dist = dist
                        best_tracker_i = i

            if best_tracker_i is not None:
                best_i = best_tracker_i

        # Strategy B: Discovery (Find the best person if tracking failed)
        if best_i is None:
            best_score = -1e9
            for i in range(len(final_kpts)):
                pose = final_kpts[i].reshape(17, 3)
                valid = self._count_valid_kpts(pose)
                if valid < 5: continue

                x1, y1, x2, y2 = final_boxes[i]
                area = (x2 - x1) * (y2 - y1)
                cx = (x1 + x2) * 0.5
                dist_to_center = abs(cx - (self.target_w * 0.5))
                score = area - (dist_to_center * 50) + (valid * 500.0)
                if score > best_score:
                    best_score = score
                    best_i = i

        if best_i is not None:
            x1, y1, x2, y2 = final_boxes[best_i]
            self.prev_center = np.array([(x1+x2)/2, (y1+y2)/2])
            self.prev_torso_len = y2 - y1
        else:
            self.prev_center = None
            self.prev_torso_len = 0

        return best_i

    def process_model_output(self, *ofmaps):
        _post_t0 = time.perf_counter()

        predict = ofmaps[0].squeeze(0).T
        predict = predict[predict[:, 4] > self.box_score, :]
        scores = predict[:, 4]
        boxes = predict[:, 0:4] / self.ratio
        boxes = self.xywh2xyxy(boxes)

        kpts = predict[:, 5:]
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3*j+2] < self.kpt_score:
                    kpts[i, 3*j: 3*(j+1)] = [-1, -1, -1]
                else:
                    kpts[i, 3*j] /= self.ratio
                    kpts[i, 3*j+1] /= self.ratio
        idxes = self.nms_process(boxes, scores, self.nms_thr)

        final_boxes = boxes[idxes, :].astype(int)
        final_kpts = kpts[idxes, :]
        current_img = self.capture_queue.get()
        self.capture_queue.task_done()

        if self.state == "ANALYZING":
            txt = f"Frame: {len(self.ref_kpts_list)}"
            cv.putText(current_img, txt, (30, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(current_img, txt, (30, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv.LINE_AA)
            self.ref_frames_cache.append(current_img.copy())

            # Target User Selection heuristic
            # In case of multiple people in the frame, we select the "active" user based on:
            # 1. Bounding box area (closer/larger is better).
            # 2. Centrality (closer to screen center is better).
            # 3. Keypoint validity (more visible joints is better).
            if len(final_kpts) > 0:
                pose = final_kpts[0].reshape(17, 3)
                self.ref_kpts_list.append(pose)
            else:
                if len(self.ref_kpts_list) > 0:
                    self.ref_kpts_list.append(self.ref_kpts_list[-1])
                else:
                    self.ref_kpts_list.append(np.zeros((17, 3)))

            self.show(current_img)
            _post_t1 = time.perf_counter()
            self._post_ms_acc += (_post_t1 - _post_t0) * 1000.0
            self._post_cnt += 1
            self._update_perf_and_maybe_print()

            return current_img

        elif self.state == "LIVE":
            if not self.live_mode_initialized or self.ref_w_final == 0:
                return current_img

            current_pose = None
            if len(final_kpts) > 0:
                pick_i = self._pick_user_idx(final_boxes, final_kpts)
                if pick_i is not None:
                    current_pose = final_kpts[pick_i].reshape(17, 3)
                    self.miss_det_cnt = 0
                else:
                    self.miss_det_cnt += 1
            else:
                self.miss_det_cnt += 1

            if current_pose is None:
                current_pose = np.zeros((17, 3), dtype=np.float32)

            self.user_buffer.append(current_pose)
            if self._count_valid_kpts(current_pose) > 5:
                 self.draw_skeleton(current_img, current_pose.tolist(), None)

            # =========================================================
            #   CORE LOGIC: Real-time Scoring & Rep Counting
            # =========================================================

            if len(self.user_buffer) == self.cfg.EVAL_WINDOW and \
               (self.frame_counter % self.cfg.EVAL_INTERVAL == 0):

                user_seq = np.array(self.user_buffer)
                curr_idx = self.playback_idx
                start_idx = curr_idx - self.cfg.EVAL_WINDOW

                if start_idx >= 0 and curr_idx < len(self.ref_kpts_list):
                    ref_seq = np.array(self.ref_kpts_list[start_idx : curr_idx])

                    if self.comparator:
                        _dtw_t0 = time.perf_counter()
                        raw, final, dist, _, _ = self.comparator.compare_pair(ref_seq, user_seq)
                        _dtw_t1 = time.perf_counter()
                        self._dtw_ms_acc += (_dtw_t1 - _dtw_t0) * 1000.0
                        self._dtw_cnt += 1
                        self.current_score = final

                        if self.miss_det_cnt > 10:
                            self.ui_status_text = "DETECTING..."
                            self.ui_status_color = (0, 0, 255) # Red
                            self.current_score = 0.0
                        elif final < self.cfg.SCORE_THRESHOLD:
                            self.ui_status_text = "MISS"
                            self.ui_status_color = (0, 165, 255) # Orange
                        else:
                            if final > self.cfg.PERFECT_SCORE:
                                self.ui_status_text = "PERFECT"
                                self.ui_status_color = (0, 255, 0) # Green
                            else:
                                self.ui_status_text = "GOOD"
                                self.ui_status_color = (0, 255, 255) # Yellow

                        # Repetition Counting with Hysteresis (Schmitt Trigger)
                        # Raw scores can be noisy/jittery. To avoid double-counting a single rep:
                        # 1. Trigger HIGH: Count the rep only when score exceeds thresholds.
                        # 2. Lock State: Set a flag (rep_trigger_active) to prevent re-counting.
                        # 3. Reset LOW: Only release the lock when score drops significantly (hysteresis).
                        if final >= self.cfg.SCORE_THRESHOLD:
                            if not self.rep_trigger_active:
                                self.rep_count += 1
                                self.rep_trigger_active = True
                        elif final < (self.cfg.SCORE_THRESHOLD - 0.1):
                            # Unlock only when score drops significantly (hysteresis)
                            # This prevents the count from jittering when score is borderline 0.7
                            self.rep_trigger_active = False

                        print(f"[EVAL] Rep={self.rep_count} Score={final:.4f} Status={self.ui_status_text}")

            self.frame_counter += 1

            # =========================================================
            #   DISPLAY LOGIC
            # =========================================================
            # Prepare Coach Frame
            if len(self.ref_frames_cache) > 0:
                safe_idx = self.playback_idx % len(self.ref_frames_cache)
                ref_img = self.ref_frames_cache[safe_idx].copy()
                if safe_idx < len(self.ref_kpts_list):
                    ref_pose = self.ref_kpts_list[safe_idx]
                    self.draw_skeleton(ref_img, ref_pose.tolist(), None)
                self.playback_idx = (self.playback_idx + 1) % len(self.ref_frames_cache)
                if self.playback_idx == 0:
                    self.user_buffer.clear()
                    self.rep_trigger_active = False # Reset trigger on loop
            else:
                ref_img = np.zeros((self.target_h, self.ref_w_final, 3), dtype=np.uint8)

            # Combine Canvas
            self.display_canvas[:, :self.ref_w_final] = ref_img
            h, w = current_img.shape[:2]
            if h == self.target_h and w == self.target_w:
                self.display_canvas[:, self.ref_w_final:] = current_img
            else:
                self.display_canvas[:, self.ref_w_final:] = cv.resize(current_img, (self.target_w, self.target_h))

            # Draw UI
            cx = self.ref_w_final
            font = cv.FONT_HERSHEY_SIMPLEX

            cv.putText(self.display_canvas, "COACH", (20, 40), font, 1.0, (0, 0, 0),     5, cv.LINE_AA)
            cv.putText(self.display_canvas, "COACH", (20, 40), font, 1.0, (255, 255, 0), 2, cv.LINE_AA)
            user_lbl = "USER"
            (u_w, u_h), _ = cv.getTextSize(user_lbl, font, 1.0, 2)
            user_x = self.display_canvas.shape[1] - u_w - 20
            cv.putText(self.display_canvas, user_lbl, (user_x, 40), font, 1.0, (0, 0, 0),     5, cv.LINE_AA)
            cv.putText(self.display_canvas, user_lbl, (user_x, 40), font, 1.0, (255, 255, 0), 2, cv.LINE_AA)

            font_scale = 1.0
            thickness = 2
            (text_w, text_h), baseline = cv.getTextSize(self.ui_status_text, font, font_scale, thickness)

            text_x = cx - text_w // 2
            text_y = 60
            cv.rectangle(self.display_canvas,
                         (text_x - 10, text_y - text_h - 10),
                         (text_x + text_w + 10, text_y + baseline + 10),
                         (40, 40, 40), -1)
            cv.putText(self.display_canvas, self.ui_status_text, (text_x, text_y),
                       font, font_scale, self.ui_status_color, thickness, cv.LINE_AA)

            # Progress Bar
            bar_width = 300
            bar_height = 15
            bar_x = cx - bar_width // 2
            bar_y = text_y + 25
            fill_width = int(bar_width * self.current_score)

            cv.rectangle(self.display_canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (80,80,80), -1)
            cv.rectangle(self.display_canvas, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), self.ui_status_color, -1)
            cv.rectangle(self.display_canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255,255,255), 2)

            # Rep Count & Score Value
            info_text = f"Reps: {self.rep_count} | Score: {self.current_score:.2f}"
            (i_w, i_h), _ = cv.getTextSize(info_text, font, 1.0, 2)
            info_x = cx - i_w // 2
            info_y = bar_y + 40
            cv.putText(self.display_canvas, info_text, (info_x, info_y), font, 1.0, (0,0,0), 8, cv.LINE_AA)
            cv.putText(self.display_canvas, info_text, (info_x, info_y), font, 1.0, (0,255,255), 2, cv.LINE_AA)

            # FPS
            fps_text = f"FPS: {self._fps_value:.1f}"
            (f_w, f_h), f_base = cv.getTextSize(fps_text, font, 1.0, 2)
            fps_x = cx - f_w // 2
            fps_y = self.display_canvas.shape[0] - 20
            cv.rectangle(self.display_canvas, (fps_x - 8, fps_y - f_h - 8), (fps_x + f_w + 8, fps_y + f_base + 6), (0, 0, 0), -1)
            cv.putText(self.display_canvas, fps_text, (fps_x, fps_y),
                       font, 0.7, (255, 255, 255), 2, cv.LINE_AA)

            self.show(self.display_canvas)

            _post_t1 = time.perf_counter()
            self._post_ms_acc += (_post_t1 - _post_t0) * 1000.0
            self._post_cnt += 1
            self._update_perf_and_maybe_print()

            return self.display_canvas

    def _update_perf_and_maybe_print(self):
        self._fps_frames += 1
        now = time.perf_counter()
        dt = now - self._fps_last_t
        if dt >= 1.0:
            self._fps_value = self._fps_frames / dt
            self._fps_frames = 0
            self._fps_last_t = now

        if self._post_cnt > 0 and (self._post_cnt % self._print_every == 0):
            pp_avg = (self._pp_ms_acc / self._pp_cnt) if self._pp_cnt > 0 else 0.0
            post_avg = (self._post_ms_acc / self._post_cnt) if self._post_cnt > 0 else 0.0
            dtw_avg = (self._dtw_ms_acc / self._dtw_cnt) if self._dtw_cnt > 0 else 0.0

            print(f"[PERF] FPS={self._fps_value:.2f} | preprocess={pp_avg:.2f} ms | postprocess={post_avg:.2f} ms | DTW={dtw_avg:.2f} ms")

    def show(self, img):
        cv.imshow('Fitness Mirror', img)
        if cv.waitKey(1) == ord('q'):
            if self.cam: self.cam.release()
            if self.live_cap: self.live_cap.release()
            cv.destroyAllWindows()
            sys.exit(1)

def run_mxa(dfp, post_model, app):
    accl = AsyncAccl(dfp)
    accl.set_postprocessing_model(post_model, model_idx=0)
    accl.connect_input(app.generate_frame)
    accl.connect_output(app.process_model_output)
    accl.wait()


def existing_file(path: str) -> str:
    # Validate a file path for argparse
    if path is None:
        raise argparse.ArgumentTypeError("path is None")
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return path


if __name__ == '__main__':
    print("Starting Pose Estimation Application...")

    parser = argparse.ArgumentParser(description="Run MX3 real-time inference")
    parser.add_argument('-d', '--dfp', type=str, default="../../models/YOLO_v8_medium_pose_640_640_3_onnx.dfp", help="Path to DFP")
    parser.add_argument('-post', '--post_model', type=str, default="../../models/YOLO_v8_medium_pose_640_640_3_onnx_post.onnx", help="Path to Post ONNX")
    parser.add_argument("-i", "--input_video", type=existing_file, default=None, help="Input coach video file (required when cache is missing and source=video)")
    parser.add_argument('-s', '--source', type=str, default='cam', choices=['cam', 'video'], help="Select input source: 'cam' (Webcam) or 'video' (Coach Video File)")
    parser.add_argument("--prefix", type=str, default="coach", help="Filename prefix for cache files (e.g., coach -> coach_action.mp4 / coach_action.npy)")
    args = parser.parse_args()
    config = Config(PREFIX=args.prefix)

    if args.source == 'cam':
        cam_source = 0
        mirror_mode = True
        print(">>> Mode: LIVE CAMERA (Webcam)")
    else:
        # Use --input_video as the reference coach video source
        if args.input_video is None and (not os.path.exists(config.FILENAME_NPY) or not os.path.exists(config.FILENAME_VID)):
            parser.error("--input_video is required when reference cache is missing and --source=video")

        cam_source = args.input_video
        mirror_mode = False
        print(f">>> Mode: COACH VIDEO (reference) = {args.input_video}")

    app = App(
        cam_source=cam_source,
        model_input_shape=(640, 640),
        mirror=mirror_mode,
        src_is_cam=(args.source == "cam"),
        cfg=config,
        input_video_path=args.input_video
    )

    print("Running MXA Inference...")
    run_mxa(args.dfp, args.post_model, app)
