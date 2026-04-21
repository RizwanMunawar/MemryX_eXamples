import time
import argparse
import numpy as np
import cv2
import os
from queue import Queue, Full
import queue
from threading import Thread
from memryx import AsyncAccl
from yolov8_seg import YoloV8Seg as YoloModel
from collections import defaultdict
import sys


FPS_LOG_INTERVAL = 30  # print out FPS every X frames


class App:
    """
    A demo app to run YOLOv8 on the MemryX MXA.
    """

    def __init__(self, video, model_type, show=True):
        """
        Initialization function.
        """
        # Display control and stream initialization
        self.show = show
        self.done = False

        # Stream-related containers and initialization
        self.cap_queue = Queue(maxsize=5)
        self.dets_queue = Queue(maxsize=5)
        self.model_type = model_type
        
        # FPS calculation related
        self.frame_count = 0
        self.start_ms = 0
        self.fps_number = 0.0
        self.history_fps = []
        self.total_frame_count = 0
        
        self.cam = cv2.VideoCapture(video)
        if "/dev/video" in video:
            self.srcs_is_cam = True
        else:
            self.srcs_is_cam = False

        self.color_wheel = np.random.randint(0, 255, (20, 3)).astype(np.int32)
        self.model = YoloModel(self.cam, model_type=self.model_type)

    def run(self):
        """
        Start inference on the MXA using multiple streams.
        """
        print("YOLOv8 inference starts...")

        if self.model_type != 'numpy':
            print ("Using DFP and Post-processing model")
            accl = AsyncAccl(dfp=self.dfp, use_model_shape=(True, True))  # Initialize the accelerator with DFP
            accl.set_postprocessing_model(self.post_model, model_idx=0)  # Set the post-processing model
        else:
            print ("Using Numpy-implementation postprocessing, no post model is used")
            accl = AsyncAccl(dfp=self.dfp, use_model_shape=(False, False))  # Initialize the accelerator with DFP

        # Connect input and output streams for the accelerator
        accl.connect_input(self.capture_and_preprocess)
        accl.connect_output(self.postprocess)
        accl.wait()

        self.done = True

    def capture_and_preprocess(self):
        """
        Captures a frame for the video device and pre-processes it.
        """

        while True:
            got_frame, frame = self.cam.read()
                
            if not got_frame or self.done:
                print("No frame or done")
                return None

            if self.srcs_is_cam and self.cap_queue.full():
                # drop frame
                continue
            else:
                # Put the frame in the cap_queue to be processed later
                self.cap_queue.put(frame)

                # Pre-process the frame using the corresponding model
                frame = self.model.preprocess(frame)
                
                # Transpose the frame based on model type
                if self.model_type == "tflite":
                    frame = frame.transpose(0, 2, 3, 1)  # (1, 3, 640, 640) -> (1, 640, 640, 3)
                elif self.model_type == "onnx":
                    pass # keep (1, 3, 640, 640)
                else:
                    frame = frame.transpose(2, 3, 0, 1)  # (1, 3, 640, 640) -> (640, 640, 1, 3)

                return frame


    def postprocess(self, *mxa_output):
        """
        Post-process the output from MXA.
        """
        dets = self.model.postprocess(mxa_output)  # Get detection results

        # Queue detection results for display
        frame = self.cap_queue.get()
        if self.show:
            self.display(dets, frame)
        
        # Calculate FPS
        self.update_fps()
        
    def update_fps(self):

        # increment frame count
        self.frame_count += 1

        now_ms = int(time.time() * 1000)

        if self.frame_count == 1:
            self.start_ms = now_ms # record start time
        else:
            # update fps_number
            duration_ms = now_ms - self.start_ms
            self.fps_number = (self.frame_count * 1000.0) / duration_ms

            # print FPS
            if self.frame_count % FPS_LOG_INTERVAL == 0:
                print(f"Frame cnt: {self.frame_count} => FPS: {self.fps_number:.2f}")

                # Update history
                self.history_fps.append(self.fps_number)


    def get_avg_fps(self):
        return np.mean(self.history_fps)

    def display(self, dets, frame):
        
        # Draw detection boxes
        seg_canvas = frame.copy()
        for d in dets:
            x1, y1, w, h = d['bbox']
            color = tuple(int(c) for c in self.color_wheel[d['class_id'] % 20])

            # Draw bounding boxes
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            # Add class label
            frame = cv2.putText(frame, d['class'], (x1 + 2, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw the contour and fill mask
            segment = d['segment']
            seg_canvas[segment] = color
          
        # Add FPS to frame
        fps_text = f"{self.fps_number:.2f}"
        frame = cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Blend segmentation canvas with original frame
        frame = cv2.addWeighted(seg_canvas, 0.3, frame, 0.7, 0)

        window_name = f"YOLOv8s"
        cv2.imshow(window_name, frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            self.done = True
        


def main(args):
    """
    Main function to start YOLOv8s inference.
    """
    if args.post_model.endswith('.onnx'):
        model_type = 'onnx'
    elif args.post_model.endswith('.tflite'):
        model_type = 'tflite'
    elif args.post_model == 'numpy':
        model_type = 'numpy'
    else:
        raise ValueError(f"Unsupported post-processing model format: {args.post_model}")

    # Initialize the application with video paths and display settings
    app = App(video=args.video, model_type=model_type, show=args.show)
    app.dfp = args.dfp  # Set the DFP path from arguments
    app.post_model = args.post_model  # Set the post-processing model path from arguments
    app.run()  # Start inference

    # Print final average FPS for each stream
    print(f'\n\nFinal Avg FPS: {app.get_avg_fps():.2f}')


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="\033[34mMemryX YoloV8 Demo\033[0m")
    
    # Video input paths
    parser.add_argument('--video', dest="video", 
                        action="store", 
                        default='/dev/video0',
                        help="Path to video files for inference. Use '/dev/video0' for webcam. (Default:'/dev/video0')")
    
    # Option to turn off display
    parser.add_argument('--no_display', dest="show", 
                        action="store_false", 
                        default=True,
                        help="Optionally turn off the video display")

    # DFP model argument
    parser.add_argument('-d', '--dfp', type=str, 
                        default="../../models/onnx/YOLO_v8_nano_seg_640_640_3_onnx.dfp", 
                        help="Path to the compiled DFP file (default: '../../models/onnx/YOLO_v8_nano_seg_640_640_3_onnx.dfp')")

    # Post-processing model argument
    parser.add_argument('-p', '--post_model', type=str, 
                        default="numpy", 
                        help="Path to the post-processing file (default: 'numpy')")

    args = parser.parse_args()
    
    # Check if the path exists
    if not os.path.exists(args.video) and args.video != '/dev/video0':
        raise FileNotFoundError(f"Video source not found: {args.video}")
    if not os.path.exists(args.dfp):
        raise FileNotFoundError(f"DFP model not found: {args.dfp}")
    if not os.path.exists(args.post_model) and args.post_model != 'numpy':
        raise FileNotFoundError(f"Post-processing model not found: {args.post_model}")

    # Call the main function
    main(args)

