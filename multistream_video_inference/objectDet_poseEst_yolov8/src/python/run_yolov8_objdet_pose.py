"""
============
Information:
============
Project: YOLOv8s example code on MXA
File Name: app.py

============
Description:
============
A script to show how to use the MultiStreamAcclerator API to perform a real-time inference
on MX3 using YOLOv8s model.
"""

###################################################################################################

# Imports
import argparse
import numpy as np
import cv2
from queue import Queue, Full
from threading import Thread
from memryx import MultiStreamAsyncAccl
from yolov8 import YoloV8 as YoloModel
from typing import List

###################################################################################################

class Yolo8sMxa:
    """
    A demo app to run YOLOv8s on the MemryX MXA.
    """

###################################################################################################
    def __init__(self, video_paths, show=True):
        """
        Initialization function.
        """
        # Display control and stream initialization
        self.show = show
        self.done = False
        self.num_streams = len(video_paths)  # Number of streams

        # Stream-related containers and initialization
        self.streams = []
        self.streams_idx = [True] * self.num_streams  # True while stream is alive
        self.stream_window = [False] * self.num_streams
        
        # Separate queues for captured frames - each model gets frames from its own queue
        self.detection_frame_queue = {i: Queue(maxsize=4) for i in range(self.num_streams)}
        self.pose_frame_queue      = {i: Queue(maxsize=4) for i in range(self.num_streams)}
        
        # Result queues
        self.dets_queue = {i: Queue(maxsize=5) for i in range(self.num_streams)}
        self.pose_queue = {i: Queue(maxsize=5) for i in range(self.num_streams)}
        
        # Display queue - stores original frames for visualization
        self.display_queue = {i: Queue(maxsize=4) for i in range(self.num_streams)}
        
        self.outputs = {i: [] for i in range(self.num_streams)}
        self.dims = {}
        self.color_wheel = {}
        self.model = {}
        self.writer = {i: None for i in range(self.num_streams)}
        self.srcs_are_cams = {i: True for i in range(self.num_streams)}

        # Predefined color list for drawing keypoints
        self.COLOR_LIST = list([
            [128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
            [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
            [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0],
            [255, 0, 51], [153, 0, 153], [51, 0, 51], [0, 0, 0],
            [0, 102, 255], [0, 51, 255], [0, 153, 255], [0, 153, 153]
        ])

        # Define keypoint pairs for drawing skeletons
        self.KEYPOINT_PAIRS = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        self.model_input_shape = (640, 640)
        self.box_score = 0.25  # Threshold for object confidence
        self.ratio = {i: 1.0 for i in range(self.num_streams)}  # Ratio for each stream
        self.kpt_score = 0.5  # Threshold for keypoint confidence
        self.nms_thr = 0.2    # IoU threshold for non-max suppression
        
        # Frame capture threads
        self.capture_threads = []

        # Initialize video captures, models, and dimensions for each stream
        for i, video_path in enumerate(video_paths):
            if "/dev/video" in video_path:
                self.srcs_are_cams[i] = True
            else:
                self.srcs_are_cams[i] = False

            vidcap = cv2.VideoCapture(video_path)
            self.streams.append(vidcap)

            # Get frame dimensions
            self.dims[i] = (
                int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            self.color_wheel[i] = np.random.randint(0, 255, (20, 3)).astype(np.int32)
            self.input_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.input_width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Initialize the YOLOv8 model
            self.model[i] = YoloModel(stream_img_size=(self.dims[i][1], self.dims[i][0], 3))

        # Start capture threads - one per stream
        for i in range(self.num_streams):
            capture_thread = Thread(target=self.capture_frames, args=(i,))
            self.capture_threads.append(capture_thread)

        # Display thread
        self.display_thread = Thread(target=self.display)

###################################################################################################
    def run(self):
        """
        Start inference on the MXA using multiple streams.
        """
        print("YOLOv8s inference on MX3 started")

        accl = MultiStreamAsyncAccl(dfp=self.dfp, use_model_shape=(False, True))
        accl.set_postprocessing_model(self.post_model, model_idx=1)

        # Start capture threads
        for thread in self.capture_threads:
            thread.start()
            
        # Start display thread
        self.display_thread.start()

        # Connect input and output streams for the accelerator
        accl.connect_streams(self.preprocess_for_detection, self.postprocess,     self.num_streams, 0)
        accl.connect_streams(self.preprocess_for_pose,       self.pose_postprocess, self.num_streams, 1)

        # Wait for the accelerator to finish (all streams must eventually return None)
        accl.wait()

        # When accelerator is done, signal global shutdown
        self.done = True

        # Join threads
        for thread in self.capture_threads:
            thread.join()
        self.display_thread.join()

###################################################################################################
    def capture_frames(self, stream_idx):
        while not self.done:
            got_frame, frame = self.streams[stream_idx].read()

            if not got_frame or self.done:
                self.streams_idx[stream_idx] = False
                break

            det_q  = self.detection_frame_queue[stream_idx]
            pose_q = self.pose_frame_queue[stream_idx]
            disp_q = self.display_queue[stream_idx]

            try:
                if self.srcs_are_cams[stream_idx]:
                    # If ANY queue is full, drop this frame for ALL of them
                    if det_q.full() or pose_q.full() or disp_q.full():
                        continue

                    f = frame.copy()
                    det_q.put(f, block=False)
                    pose_q.put(f.copy(), block=False)
                    disp_q.put(frame, block=False)
                else:
                    # Video: block so we don't outrun processing
                    f = frame.copy()
                    det_q.put(f, timeout=2)
                    pose_q.put(f.copy(), timeout=2)
                    disp_q.put(frame, timeout=2)
            except Full:
                continue


###################################################################################################
    def preprocess_for_detection(self, stream_idx):
        """
        Gets a frame from detection queue and pre-processes it for object detection.
        """
        while True:

            if self.done or not self.streams_idx[stream_idx]:
                return None

            try:
                frame = self.detection_frame_queue[stream_idx].get(timeout=2)

                if self.done or not self.streams_idx[stream_idx]:
                    return None

                # Pre-process the frame using the YOLOv8 model
                frame = self.model[stream_idx].preprocess(frame)
                return frame

            except Exception:
                if self.done or not self.streams_idx[stream_idx]:
                    return None
                continue

    def preprocess_for_pose(self, stream_idx):
        """
        Gets a frame from pose queue and pre-processes it for pose estimation.
        """
        while True:
            if self.done or not self.streams_idx[stream_idx]:
                return None

            try:
                frame = self.pose_frame_queue[stream_idx].get(timeout=2)
                if self.done or not self.streams_idx[stream_idx]:
                    return None

                # Pre-process the frame for pose estimation
                h, w = frame.shape[:2]
                self.ratio[stream_idx] = min(
                    self.model_input_shape[0] / h,
                    self.model_input_shape[1] / w
                )
                image_resized = cv2.resize(
                    frame,
                    (int(w * self.ratio[stream_idx]), int(h * self.ratio[stream_idx])),
                    interpolation=cv2.INTER_LINEAR
                )

                # Create a padded image
                padded_img = np.ones(
                    (self.model_input_shape[0], self.model_input_shape[1], 3),
                    dtype=np.uint8
                ) * 114
                padded_img[:int(h * self.ratio[stream_idx]), :int(w * self.ratio[stream_idx])] = image_resized

                # Normalize image to [0, 1] range
                padded_img = padded_img / 255.0
                padded_img = padded_img.astype(np.float32)
                
                # Keep as (640, 640, 3) for numpy mode
                return padded_img

            except Exception:
                if self.done or not self.streams_idx[stream_idx]:
                    return None
                continue

###################################################################################################
    def postprocess(self, stream_idx, *mxa_output):
        """
        Post-process the output from MXA for object detection.
        """
        if self.done:
            return

        dets = self.model[stream_idx].postprocess(mxa_output)  # Get detection results

        try:
            self.dets_queue[stream_idx].put(dets, block=False)
        except Full:
            pass

    def xywh2xyxy(self, box: np.ndarray) -> np.ndarray:
        # Convert bounding boxes from [x, y, w, h] format to [x1, y1, x2, y2] format
        box_xyxy = box.copy()
        box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
        box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
        box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
        box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
        return box_xyxy

    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        box and boxes are in format [x1, y1, x2, y2]
        """
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

    def pose_postprocess(self, stream_idx, *ofmaps):
        """Post-process pose output from MXA for one stream."""
        if self.done:
            return

        # 1) Process model output (keypoints and bounding boxes)
        predict = ofmaps[0].squeeze(0).T  # → [8400, 56] if ofmaps[0] is (1, 56, 8400)

        predict = predict[predict[:, 4] > self.box_score, :]
        scores  = predict[:, 4]
        boxes   = predict[:, 0:4] / self.ratio[stream_idx]

        boxes = self.xywh2xyxy(boxes)

        # Keypoints
        kpts = predict[:, 5:]
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3*j+2] < self.kpt_score:
                    kpts[i, 3*j:3*(j+1)] = [-1, -1, -1]
                else:
                    kpts[i, 3*j]   /= self.ratio[stream_idx]
                    kpts[i, 3*j+1] /= self.ratio[stream_idx]

        idxes = self.nms_process(boxes, scores, self.nms_thr)
        result = {
            "boxes":  boxes[idxes, :].astype(int).tolist()   if len(idxes) > 0 else [],
            "kpts":   kpts[idxes, :].astype(float).tolist()   if len(idxes) > 0 else [],
            "scores": scores[idxes].tolist()                  if len(idxes) > 0 else [],
        }

        # Non-blocking put: drop if queue is full to avoid deadlock
        try:
            self.pose_queue[stream_idx].put(result, block=False)
        except Full:
            pass

###################################################################################################
    def display(self):
        while not self.done:
            for stream_idx in range(self.num_streams):
                if (not self.display_queue[stream_idx].empty()
                    and not self.dets_queue[stream_idx].empty()
                    and not self.pose_queue[stream_idx].empty()):  

                    frame = self.display_queue[stream_idx].get()
                    dets  = self.dets_queue[stream_idx].get()
                    pose_result = self.pose_queue[stream_idx].get()

                    self.display_queue[stream_idx].task_done()
                    self.dets_queue[stream_idx].task_done()
                    self.pose_queue[stream_idx].task_done()

                    # draw detections
                    for d in dets:
                        x1, y1, w, h = d['bbox']
                        color = tuple(int(c) for c in self.color_wheel[stream_idx][d['class_id'] % 20])
                        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
                        frame = cv2.putText(frame, d['class'], (x1 + 2, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # draw pose (boxes_out, kpts_out, scores_out = from pose_result)
                    boxes_out, kpts_out, scores_out = (
                        pose_result["boxes"],
                        pose_result["kpts"],
                        pose_result["scores"],
                    )
                    for kpt, score in zip(kpts_out, scores_out):
                        for pair in self.KEYPOINT_PAIRS:
                            pt1 = kpt[3 * pair[0] : 3 * (pair[0] + 1)]
                            pt2 = kpt[3 * pair[1] : 3 * (pair[1] + 1)]
                            if pt1[2] > 0 and pt2[2] > 0:
                                cv2.line(
                                    frame,
                                    (int(pt1[0]), int(pt1[1])),
                                    (int(pt2[0]), int(pt2[1])),
                                    (255, 255, 255),
                                    2,
                                )
                        for idx in range(len(kpt) // 3):
                            x, y, sc = kpt[3*idx : 3*(idx+1)]
                            if sc > 0:
                                cv2.circle(
                                    frame,
                                    (int(x), int(y)),
                                    4,
                                    self.COLOR_LIST[idx % len(self.COLOR_LIST)],
                                    -1,
                                )

                    if self.show:
                        window_name = f"Stream {stream_idx} - YOLOv8s + Pose"
                        cv2.imshow(window_name, frame)

            if cv2.waitKey(1) == ord('q'):
                self.done = True

        cv2.destroyAllWindows()
        for stream in self.streams:
            stream.release()

###################################################################################################

def main(args):
    """
    Main function to start YOLOv8s inference.
    """
    # Initialize the application with video paths and display settings
    yolo8s_inf = Yolo8sMxa(video_paths=args.video_paths, show=args.show)
    yolo8s_inf.dfp = args.dfp          # Set the DFP path from arguments
    yolo8s_inf.post_model = args.post_model  # Set the post-processing model path from arguments
    yolo8s_inf.run()  # Start inference

###################################################################################################

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="\033[34mMemryX YoloV8s Demo\033[0m")
    
    # Video input paths
    parser.add_argument(
        '--video_paths',
        nargs='+',
        dest="video_paths", 
        action="store", 
        default=['/dev/video0'],
        help="Path to video files for inference. Use '/dev/video0' for webcam. (Default:'/dev/video0')"
    )
    
    # Option to turn off display
    parser.add_argument(
        '--no_display',
        dest="show", 
        action="store_false", 
        default=True,
        help="Optionally turn off the video display"
    )

    # DFP model argument
    parser.add_argument(
        '-d', '--dfp',
        type=str, 
        default='../../models/models.dfp', 
        help="Path to the compiled DFP file (default: 'models/models.dfp')"
    )
    
    # Post-processing model argument
    parser.add_argument(
        '-p', '--post_model',
        type=str, 
        default='../../models/model_1_yolov8n-pose_post.onnx', 
        help="Path to the post-processing ONNX file (default: 'models/model_1_yolov8n-pose_post.onnx')"
    )

    args = parser.parse_args()

    # Call the main function
    main(args)

# eof
