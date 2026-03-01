from PyQt5.QtCore import (
    QThread, pyqtSignal, QCoreApplication, QObject, QTimer, pyqtSlot, Qt
)
from queue import Queue, Empty
import numpy as np
import argparse
import time
import cv2
import sys
import os

from memryx import AsyncAccl
from helper_classes import (
    Annotator,
    YOLOV8Postprocessor,
    CameraWorker,
    SpeedDistanceEstimatorWorker,
    SortTrackerPipeline,
    compute_performance_measures,
    correct_boxes_for_camera
)

# ================================
# Display Controller (MAIN THREAD)
# - imshow + waitKey ONLY here
# - Reads frames from a shared Queue (FIFO)
# - Exits only after it receives EOS sentinel (None)
# ================================
class DisplayController(QObject):
    request_stop = pyqtSignal()  # user pressed 'q'

    def __init__(self, fps: float, display_queue: Queue, window_name="FootballCV - MemryX", width=1920, height=1080):
        super().__init__()

        if fps <= 0:
            fps = 30.0

        self.fps = float(fps)
        self.period_s = 1.0 / self.fps
        self.q = display_queue
        self.window_name = window_name

        self._eos_received = False
        self._next_due = time.perf_counter()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        # Single-shot timer so we can schedule precisely
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)
        self._schedule(0)

    def _schedule(self, ms: int):
        self._timer.start(max(0, int(ms)))

    def _tick(self):
        # Always pump OpenCV events & read keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.request_stop.emit()

        now = time.perf_counter()

        # Respect FPS pacing: don't display before next due time
        if now < self._next_due:
            self._schedule(int((self._next_due - now) * 1000))
            return

        # If EOS already received, we can quit cleanly
        if self._eos_received:
            cv2.destroyAllWindows()
            QCoreApplication.instance().quit()
            return

        # Try to get next frame (FIFO). If none available, poll again soon.
        try:
            item = self.q.get_nowait()
        except Empty:
            self._schedule(1)
            return

        # Sentinel means no more frames will arrive
        if item is None:
            self._eos_received = True
            self._schedule(0)
            return

        frame = item

        # Ensure OpenCV-friendly buffer
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        frame = np.ascontiguousarray(frame)

        cv2.imshow(self.window_name, frame)

        # Set next due time (locks playback to FPS)
        now2 = time.perf_counter()
        self._next_due = now2 + self.period_s
        self._schedule(int(self.period_s * 1000))


# ================================
# Accelerator Thread
# ================================
class InferenceThread(QThread):
    finished_signal = pyqtSignal()
    frame_ready_signal = pyqtSignal(np.ndarray, dict, tuple)

    def __init__(self, video_path, dfp_path, post_model): 
        super().__init__()
        
        self.cap = cv2.VideoCapture(video_path)
        # This queue to store original frame to combine them with their results.
        self.frame_queue = Queue()

        # MX3.
        self.accl = AsyncAccl(dfp_path)

        self.accl.set_postprocessing_model(post_model)
        self.accl.connect_input(self.data_source)
        self.accl.connect_output(self.output_processor)

        # This post processing to process bboxes and scores from yolo.
        self.postprocessor = YOLOV8Postprocessor(conf_thres=0.55, iou_thres=0.5)

        # Camera Worker for Optical Flow. This worker runs on a separate thread.
        self.camera_worker = CameraWorker()

        # A Tracking Algorithm.
        self.tracker_pipeline = SortTrackerPipeline(max_age=60)

        # Worker for Speed & Distance Estimation
        self.speed_distance_worker = SpeedDistanceEstimatorWorker(
            frame_rate=25,
            frame_window=5,
        )

        # Start Camera worker thread.
        self.camera_worker.start()

    def preprocess(self, frame):
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
        return np.ascontiguousarray(img)

    def data_source(self):        
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            self.camera_worker.add_frame(frame) # Start calculating camera movements early
            self.frame_queue.put(frame) # Store original frame.

            yield self.preprocess(frame)
        
        self.cap.release()

    def output_processor(self, *post_out):
        original_frame = self.frame_queue.get() # Get original frame to send at the end.

        # Apply post processing (NMS) on scores and bboxes.
        det = self.postprocessor(post_out[0])
        boxes, confs, cls_ids = det["boxes"], det["conf"], det["cls"]
        
        # Calculate camera's movement
        camera_movement = self.camera_worker.get_result()

        # Applying tracker.
        tracked_results = self.tracker_pipeline.process(boxes, confs, cls_ids)
        if len(tracked_results["id"]) == 0: # Exit early if empty frame of tracked_results data.
            time_stamps.append(time.perf_counter() * 1000)
            self.frame_ready_signal.emit(original_frame, tracked_results, camera_movement)
            return

        # Correct boxes for camera movement before speed/distance estimation
        results = correct_boxes_for_camera(tracked_results, camera_movement[:2])
        
        # Speed and Distance Estimator
        results_with_speed_distance = self.speed_distance_worker.process(results)

        # Correct boxes back after speed/distance estimation
        final_results = correct_boxes_for_camera(results_with_speed_distance, (-camera_movement[0], -camera_movement[1]))

        # Add final result and emit the signal for ready frames. 
        time_stamps.append(time.perf_counter() * 1000) 
        self.frame_ready_signal.emit(original_frame, final_results, camera_movement)
        
    
    # --------------------------------
    # To Run the thread
    # --------------------------------
    def run(self):
        self.accl.wait()
        self.stop()
        self.finished_signal.emit()

    def stop(self):
        self.camera_worker.stop()
        self.accl.shutdown()

# ================================
# Pipeline Worker (SEPARATE THREAD)
# - annotation + writer
# - pushes annotated frames into shared display queue (FIFO)
# - pushes EOS sentinel (None) when fully done
# ================================
class VideoPipelineWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, fps, inference_thread, display_queue: Queue, writer=None):
        super().__init__()
        self.inference_thread = inference_thread
        self.display_queue = display_queue

        self._stop_requested = False

        self.fps = fps
        self.writer = writer

        self.annotator = Annotator(INPUT_SIZE, CLASS_NAMES)

        self.current_frame = 0

    def setup_connections(self):
        self.inference_thread.frame_ready_signal.connect(
            self.draw_frame, type=Qt.QueuedConnection
        )
        self.inference_thread.finished_signal.connect(
            self.thread_finished, type=Qt.QueuedConnection
        )

    def drain_queue(self):
        try:
            while True:
                self.display_queue.get_nowait()
        except Empty:
            pass

    def request_stop(self):
        self._stop_requested = True
        self.drain_queue()
        self.inference_thread.stop()

    def split_results_by_class(self, results):

        if len(results.get('cls', [])) == 0: return {}

        cls_ids = results['cls'].astype(int)

        result_by_class = {}

        for cls_id in np.unique(cls_ids):
            mask = (cls_ids == cls_id)

            if np.sum(mask) == 0: continue

            result_by_class[int(cls_id)] = {k: v[mask] for k, v in results.items() if k in ['boxes', 'conf', 'cls', 'id', 'speed', 'distance']}

        return result_by_class

    @pyqtSlot(np.ndarray, dict, tuple)
    def draw_frame(self, frame, det, camera_movements):
        if self._stop_requested:
            return

        elapsed_ms = -1
        # ---------- Throughput FPS using ALL past frames ----------
        if len(time_stamps) >= 2:
            total_frames = len(time_stamps) - 1
            elapsed_ms = time_stamps[-1] - time_stamps[0]

        if elapsed_ms > 0:
            throughput_fps = (total_frames * 1000.0) / elapsed_ms
        else:
            throughput_fps = 0


        det_cls_split = self.split_results_by_class(det)

        x_mv, y_mv, cum_x, cum_y = camera_movements
        
        annotated = self.annotator.draw_annotations(frame, det_cls_split)
        
        h_img, w_img = annotated.shape[:2]

        # ---------- Top-left: Camera information ----------
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (600, 75), (255, 255, 255), cv2.FILLED)
        annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)

        cv2.putText(
            annotated, f"Camera Movement: X={-x_mv:.2f}, Y={-y_mv:.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        cv2.putText(
            annotated, f"Camera Displacement: X={-cum_x:.2f}, Y={-cum_y:.2f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 50, 50), 2
        )

        # ---------- Bottom-left: FPS information ----------
        fps_box_h = 70
        y0 = h_img - fps_box_h

        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, y0), (600, h_img), (255, 255, 255), cv2.FILLED)
        annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)

        cv2.putText(
            annotated, f"Video FPS: {self.fps:.2f}",
            (10, h_img - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        cv2.putText(
            annotated, f"Inference FPS: {throughput_fps:.2f}",
            (10, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )

        if self.writer:
            self.writer.write(annotated)

        # Push to display queue IN ORDER (this may block, which is OK in this worker thread)
        # Copy to avoid any accidental mutation issues
        self.display_queue.put(annotated.copy())

        self.current_frame += 1

    @pyqtSlot()
    def thread_finished(self):
        print("Inference finished; pipeline finalizing...")

        if self.writer:
            self.writer.release()

        # cap may not exist in this worker (it's commented out above)
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

        # VERY IMPORTANT: push EOS sentinel so display keeps running until it consumes it
        self.display_queue.put(None)
        self.finished.emit()


# ================================
# Main
# ================================
if __name__ == "__main__":
    global INPUT_SIZE, CLASS_NAMES, time_stamps
    INPUT_SIZE = 640
    CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]
    time_stamps = []

    parser = argparse.ArgumentParser(description="Football Video Inference with MemryX")
    parser.add_argument("--dfp", type=str, default="../models/footballcv_v8s_640_640_3.dfp",
                        help="Path to the DFP model file (default: ../models/footballcv_v8s_640_640_3.dfp)")
    parser.add_argument("--post", type=str, default="../models/footballcv_v8s_640_640_3_post.onnx",
                        help="Path to the post-processing model file (default: ../models/footballcv_v8s_640_640_3_post.onnx)")
    parser.add_argument("--video", type=str, default="../assets/video.mp4",
                        help="Path to the input video file (default: ../assets/video.mp4)")
    parser.add_argument("--output", type=str, default="../output/result.mp4",
                        help="Path to the output video file (default: ../output/result.mp4)")
    parser.add_argument("--display_queue", type=int, default=300,
                        help="Max frames buffered for display (avoid unbounded RAM).")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output video (default: disabled)")

    args = parser.parse_args()

    dfp_path = args.dfp
    post_model = args.post
    video_path = args.video
    output_path = args.output

    # Saving-related: get video metadata for writer and for pipeline/display fps
    # --------------------------------
    # Validate input video file
    # --------------------------------
    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found: '{video_path}'. "
            f"Please check the path and try again."
        )

    cap_meta = cv2.VideoCapture(video_path)

    if not cap_meta.isOpened():
        raise RuntimeError(
            f"Failed to open video file: '{video_path}'. "
            f"The file may be corrupted or in an unsupported format."
        )
    fps = cap_meta.get(cv2.CAP_PROP_FPS)

    w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_meta.release()

    writer = None

    if args.save:
        # Ensure output folder exists
        output_folder = os.path.dirname(output_path)
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")

        # Avoid overwriting existing video
        base, ext = os.path.splitext(output_path)
        counter = 1
        while os.path.exists(output_path):
            output_path = f"{base}_{counter}{ext}"
            counter += 1

        print(f"Saving to: {output_path}")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )
    else:
        print("Output video saving disabled (use --save to enable).")

    start_time = time.time_ns()
    app = QCoreApplication(sys.argv)

    # Shared FIFO queue between pipeline thread (producer) and main thread (consumer)
    display_q = Queue(maxsize=args.display_queue)

    # Inference thread
    inference_thread = InferenceThread(video_path, dfp_path, post_model)

    # Pipeline worker in its own thread
    pipeline_thread = QThread()
    pipeline_worker = VideoPipelineWorker(fps, inference_thread, display_q, writer=writer)
    pipeline_worker.moveToThread(pipeline_thread)

    # Display controller in MAIN thread
    display = DisplayController(
        fps=pipeline_worker.fps,
        display_queue=display_q,
    )

    # If user presses q: stop inference + pipeline
    display.request_stop.connect(pipeline_worker.request_stop)

    # Stop pipeline thread after it finishes (display will quit on EOS)
    pipeline_worker.finished.connect(pipeline_thread.quit)

    def on_pipeline_thread_started():
        pipeline_worker.setup_connections()
        inference_thread.start()

    pipeline_thread.started.connect(on_pipeline_thread_started)
    pipeline_thread.start()

    exit_code = app.exec_()

    # Cleanup threads
    pipeline_thread.wait()

    compute_performance_measures(time_stamps, start_time, time.time_ns())
    sys.exit(exit_code)