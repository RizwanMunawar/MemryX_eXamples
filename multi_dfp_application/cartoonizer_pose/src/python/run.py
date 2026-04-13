import sys
import argparse
import threading
import time
import cv2 as cv
from queue import Queue  # Standard threading queue
from PyQt5.QtWidgets import QApplication

# Memryx and App imports
from memryx import AsyncAccl, SchedulerOptions, ClientOptions
from apps import Cartoonizer, PoseEstmiation
from displayer import Displayer

def parse_args():
    parser = argparse.ArgumentParser(description="Cartoonizer and Pose Estimation demo with Multi-DFP (Threaded)")
    parser.add_argument('--video', type=str, help="Path to video file (instead of cam)")
    parser.add_argument('--frame_limit', '-f', type=int, default=10, help="Number of frames to process before swapping out.")
    parser.add_argument('--dfp_cartoon', type=str, default='../../models/Facial_cartoonizer_512_512_3_onnx.dfp',
                        help="Path to the compiled DFP file for Cartoonizer")
    parser.add_argument('--dfp_pose', type=str, default='../../models/YOLO_v8_small_pose_640_640_3_onnx.dfp',
                        help="Path to the compiled DFP file for Pose Estimation")
    parser.add_argument('--post', '-post', type=str, default='../../models/YOLO_v8_small_pose_640_640_3_onnx_post.onnx',
                        help="Path to the ONNX post-processing model for Pose Estimation")
    return parser.parse_args()

def shared_capture_loop(src, queue1, queue2, stop_flag, src_is_cam):
    cap = cv.VideoCapture(src)

    while not stop_flag.is_set():
        
        ok, frame = cap.read()

        if not ok:
            break
        
        if not src_is_cam:
            time.sleep(0.02) # slow down for video files to simulate real-time capture
        
        for q in [queue1, queue2]:
            if not q.full():
                q.put(frame.copy())

    print("[CAPTURE] Thread exiting...")
    
    cap.release()
    
    if not src_is_cam:
        stop_flag.set()
        
def run_cartoonizer(queue, dfp_path, display_fn, src_is_cam, frame_limit, stop_flag):
    sched_opts = SchedulerOptions()
    sched_opts.frame_limit = frame_limit
    client_opts = ClientOptions(True, 30.0) 
    
    accl = AsyncAccl(
        dfp_path,
        use_model_shape=(True, True),
        scheduler_options=sched_opts,
        client_options=client_opts
    )
    
    app = Cartoonizer(queue, display_fn, src_is_cam=src_is_cam, stop_flag=stop_flag)
    accl.connect_input(app.get_frame)
    accl.connect_output(app.process_model_output)

    return accl

def run_pose_estimation(queue, dfp_path, pose_post_model, input_shape, display_fn, src_is_cam, frame_limit, stop_flag):
    sched_opts = SchedulerOptions()
    sched_opts.frame_limit = frame_limit
    client_opts = ClientOptions(True, 30.0)
    
    accl = AsyncAccl(
        dfp_path,
        use_model_shape=(True, True),
        scheduler_options=sched_opts,
        client_options=client_opts
    )
    
    app = PoseEstmiation(queue, display_fn, input_shape, mirror=True,
                         src_is_cam=src_is_cam, stop_flag=stop_flag)

    accl.set_postprocessing_model(pose_post_model, model_idx=0)
    accl.connect_input(app.generate_frame)
    accl.connect_output(app.process_model_output)

    return accl

def main():
    args = parse_args()

    # GUI must run in the main thread
    qt_app = QApplication(sys.argv)
    
    # Start displayer
    displayer = Displayer()
    displayer.show()

    if args.video:
        input_source = args.video
        src_is_cam = False
    else:
        input_source = 0 
        src_is_cam = True

    # Standard threading queues and event
    queue_cartoonizer = Queue(maxsize=30)
    queue_pose = Queue(maxsize=30)
    stop_flag = threading.Event()

    # Initialize Memryx Accelerators
    accl_cartoon = run_cartoonizer(queue_cartoonizer, args.dfp_cartoon, displayer.update_left,
                                  src_is_cam, args.frame_limit, stop_flag)

    accl_pose = run_pose_estimation(queue_pose, args.dfp_pose, args.post, (640, 640),
                                    displayer.update_right, src_is_cam, args.frame_limit, stop_flag)

    # Start Capture Thread
    capture_thread = threading.Thread(
        target=shared_capture_loop,
        args=(input_source, queue_cartoonizer, queue_pose, stop_flag, src_is_cam),
        daemon=True # Daemon threads exit when main thread exits
    )
    
    print("[MAIN] Starting capture thread and inference engines...")
    capture_thread.start()

    try:
        sys.exit(qt_app.exec_())
    finally:
        print("[MAIN] App shutting down, signaling threads to stop...")
        stop_flag.set()
        
        # Give the hardware and threads a moment to drain
        accl_cartoon.wait()
        accl_pose.wait()
        
        # Joining the capture thread
        capture_thread.join(timeout=2.0)
        print("[MAIN] Cleanup complete.")

        qt_app.quit()

if __name__ == '__main__':
    main()