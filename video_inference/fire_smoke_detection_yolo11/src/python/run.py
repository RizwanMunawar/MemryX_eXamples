"""
============
Information:
============
Project: Fire and Smoke Detection example code on MXA
File Name: run.py
Author :Abdulrahman Hamza (Hamza)
============
Description:
============
A script to show how to use the mxapi MxAccl API to perform real-time inference
on MX3 using a YOLOv11n Fire/Smoke Detection model.
"""

###############################################################################
# Imports ##################################################
###############################################################################

import time
import argparse
import numpy as np
import cv2
from queue import Queue
from threading import Thread
import sys
from memryx import mxapi
from fire_smoke_model import FireSmokeModel 


###############################################################################

class FireSmokeDetectionApp:
    """
    A demo app to run Fire/Smoke Detection on the MemryX MXA.
    """

    ###############################################################################

    def __init__(self, video_path, dfp_path, postmodel_path, show=True, 
                 confidence_thres=0.25, iou_thres=0.45, display_size=None):
        self.show = show
        self.done = False
        self.dfp_path = dfp_path
        self.postmodel_path = postmodel_path
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.display_size = display_size  
        self.num_frames = 0
        self.cap_queue = Queue(maxsize=4)
        self.dets_queue = Queue(maxsize=5)
        self.gui_callback = None  
        
        # Parse video source
        if "/dev/video" in str(video_path) or video_path == 'cam' or (isinstance(video_path, str) and video_path.isdigit()):
            if video_path == 'cam':
                self.video_path = 0
            elif isinstance(video_path, str) and video_path.isdigit():
                self.video_path = int(video_path)
            else:
                self.video_path = video_path
            self.src_is_cam = True
        else:
            self.video_path = video_path
            self.src_is_cam = False
            
        # Open video capture
        self.vidcap = cv2.VideoCapture(self.video_path)
        if not self.vidcap.isOpened():
            print(f"Error: Could not open video source {video_path}")
            sys.exit(1)
            
        self.dims = (
            int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        self.fire_color = (0, 0, 255) 
        self.smoke_color = (255, 165, 0)  
        
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = 0

        self.model = FireSmokeModel(
            stream_img_size=(self.dims[1], self.dims[0], 3),
            confidence_thres=self.confidence_thres,
            iou_thres=self.iou_thres,
            post_model_path=self.postmodel_path,
        )

        self.dt_index = 0
        self.frame_end_time = 0
        self.fps = 0
        self.dt_array = np.zeros([30])
        self.fire_count = 0
        self.smoke_count = 0
        
        # Alert tracking (one alert per detection event)
        self.fire_alert_sent = False
        self.smoke_alert_sent = False
        
        self.display_thread = Thread(target=self.display, args=(), daemon=True)

    ###############################################################################
    def run(self):
        """
        The function that starts the inference on the MXA.
        """
        print("dfp path = ", self.dfp_path)
        accl = mxapi.MxAccl(
            dfp_path=self.dfp_path,
            local_mode=True,
            use_model_shape=[False, False],
        )
        print("Fire/Smoke Detection inference on MX3 started")

        # Start display thread
        self.display_thread.start()

        start_time = time.time()

        accl.connect_stream(
            self.capture_and_preprocess,
            self.postprocess,
            stream_id=0,
            model_id=0,
        )

        accl.start()
        accl.wait()
        
        self.done = True

        self.display_thread.join()
        
      
        if self.vidcap is not None:
            self.vidcap.release()
        
        running_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Total running time: {running_time:.1f}s for {self.num_frames} frames")
        if running_time > 0:
            print(f"Average FPS: {self.num_frames / running_time:.1f}")
        print(f"Fire detections: {self.fire_count}")
        print(f"Smoke detections: {self.smoke_count}")
        print(f"{'='*50}")

    
    def capture_and_preprocess(self, stream_id=0):
        """
        Captures a frame from the video device and pre-processes it.
        
        Returns:
            Preprocessed frame ready for inference, or None if no frame available
        """
        while True:
            if self.done:
                return None
            
            if not self.src_is_cam:
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                self.last_frame_time = time.time()

            if self.done:
                return None
            
            got_frame, frame = self.vidcap.read()

            if not got_frame:
                if not self.src_is_cam:
                    # For video files, loop back to beginning
                    self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    got_frame, frame = self.vidcap.read()
                    if not got_frame:
                        return None
                else:
                    return None

            if self.src_is_cam and self.cap_queue.full():
                continue
            else:
                self.num_frames += 1
                self.cap_queue.put(frame)
                frame = self.model.preprocess(frame)
                return frame

    ###############################################################################
    
    def postprocess(self, mxa_output, stream_id=0):
        """
        Post-process the MXA output.
        """
        if self.done:
            return
        if mxa_output is None:
            return

        if isinstance(mxa_output, (tuple, list)):
            model_outputs = tuple(mxa_output)
        else:
            model_outputs = (mxa_output,)
        
        # Post-process the MXA output
        dets = self.model.postprocess(model_outputs)

        # Push the results to the queue for the display thread
        if not self.done:
            self.dets_queue.put(dets)

        # Calculate current FPS
        self.dt_array[self.dt_index] = time.time() - self.frame_end_time
        self.dt_index += 1
        
        if self.dt_index % 15 == 0:
            self.fps = 1 / np.average(self.dt_array)
            if self.dt_index >= 30:
                self.dt_index = 0
            # Emit FPS update to GUI
            if self.gui_callback:
                self.gui_callback.fps_update.emit(self.fps)
        
        self.frame_end_time = time.time()

    ###############################################################################
    
    def display(self):
        """
        Draws detection boxes over the original image and displays them.
        """
        while not self.done:
            try:
                frame = self.cap_queue.get(timeout=0.1)
                dets = self.dets_queue.get(timeout=0.1)
                self.cap_queue.task_done()
                self.dets_queue.task_done()
            except:
                if self.done:
                    break
                continue

            # Count detections for this frame
            frame_fire_count = 0
            frame_smoke_count = 0

            # Draw detection boxes
            for d in dets:
                l, t, r, b = d['bbox']
                class_name = d['class']
                score = d['score']
                
                if class_name == 'fire':
                    color = self.fire_color
                    self.fire_count += 1
                    frame_fire_count += 1
                    if self.gui_callback and not self.fire_alert_sent:
                        timestamp = time.strftime("%H:%M:%S")
                        self.gui_callback.alert.emit(class_name, timestamp, score)
                        self.fire_alert_sent = True
                elif class_name == 'smoke':
                    color = self.smoke_color
                    self.smoke_count += 1
                    frame_smoke_count += 1
                    if self.gui_callback and not self.smoke_alert_sent:
                        timestamp = time.strftime("%H:%M:%S")
                        self.gui_callback.alert.emit(class_name, timestamp, score)  
                        self.smoke_alert_sent = True
                else:
                    color = (0, 255, 0)  
                
                frame = cv2.rectangle(frame, (l, t), (r, b), color, 4)
                
                label = f"{class_name}: {score:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                frame = cv2.rectangle(
                    frame, (l, t - label_h - 10), (l + label_w + 4, t), color, -1
                )

                frame = cv2.putText(
                    frame, label, (l + 2, t - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )

            if frame_fire_count == 0:
                self.fire_alert_sent = False
            if frame_smoke_count == 0:
                self.smoke_alert_sent = False

          
            if self.display_size is not None:
                frame = cv2.resize(frame, self.display_size, interpolation=cv2.INTER_LINEAR)

            # Emit detection counts to GUI
            if self.gui_callback:
                self.gui_callback.detection_update.emit(frame_fire_count, frame_smoke_count)
                self.gui_callback.frame_ready.emit(frame)


            if self.show and not self.gui_callback:
                cv2.imshow('Fire and Smoke Detection - MemryX', frame)

                # Exit on 'q' key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting... please wait")
                    self.done = True
                    break
        
        # Cleanup display resources
        if self.show and not self.gui_callback:
            cv2.destroyAllWindows()

###############################################################################

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="\033[34mFire and Smoke Detection on MemryX MX3\033[0m"
    )
    
    parser.add_argument(
        '-d', '--dfp', 
        type=str, 
        default="../../models/yolov11n_fire_smoke_detection.dfp", 
        help="Path to the compiled DFP file (default: '../../models/yolov11n_fire_smoke_detection.dfp')"
    )
    
    parser.add_argument(
        '-m', '--postmodel', 
        type=str, 
        default="../../models/yolov11n_fire_smoke_detection_post.onnx", 
        help="Path to the post-processing ONNX model (default: '../../models/yolov11n_fire_smoke_detection_post.onnx')"
    )

    parser.add_argument(
        '--video_path', 
        dest="video_path", 
        default='../../assets/fire_smoke2.mp4',
        help="Path to video file or 'cam' for webcam ('../../assets/test*.mp4')"
    )
 
    parser.add_argument(
        '--display_size',
        type=str,
        default=None,
        help='Display window size as WIDTHxHEIGHT (e.g., 1920x1080, 1280x720). Default: original video size'
    )
    
    parser.add_argument(
        '--nms',
        type=float,
        default=0.5,
        help='NMS/IoU threshold for filtering overlapping boxes (default: 0.45)'
    )

    args = parser.parse_args()
    
    # Launch GUI mode
    from PyQt6.QtWidgets import QApplication  # type: ignore
    from gui import FireSmokeGUI  # type: ignore
    
    print("Launching Fire/Smoke Detection GUI...")
    
    # Parse display size
    display_size = None
    if args.display_size:
        try:
            w, h = args.display_size.lower().split('x')
            display_size = (int(w), int(h))
        except ValueError:
            print(f"Invalid display size: {args.display_size}")
    
    # Create app instance (without running accelerator)
    app_core = FireSmokeDetectionApp(
        video_path=args.video_path,
        dfp_path=args.dfp,
        postmodel_path=args.postmodel,
        show=True,
        confidence_thres=0.25,
        iou_thres=args.nms,
        display_size=display_size
    )
    
    # Start display thread for processing
    app_core.display_thread.start()
    
    # Create Qt Application and GUI
    qt_app = QApplication(sys.argv)
    window = FireSmokeGUI(app_core, args.dfp, args.postmodel)
    window.show()
    sys.exit(qt_app.exec())
