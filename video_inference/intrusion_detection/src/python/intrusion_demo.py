"""
============
Information:
============
Project: Intrusion Detection using ByteTrack and YOLOv8m example code on MXA
File Name:  intrusion_demo.py
"""

###################################################################################################

# Imports
import argparse
import numpy as np
import cv2
from queue import Queue,Full
from memryx import AsyncAccl
from yolov8.yolov8 import YoloV8 as YoloModel
from yolov8.tracker.byte_tracker import BYTETracker
from yolov8.timer import Timer
import torch
from yolov8.visualize import plot_tracking


DEFAULT_ROI = None  # Will be calculated based on video dimensions

ALERT_NUM_FRAMES = 60


# Any object detected during this period is considered "known" and won't trigger an alert
# After this period, only NEW objects entering the ROI will be flagged as intruders
INIT_NUM_FRAMES = 50

###################################################################################################
###################################################################################################
###################################################################################################

def make_parser():
    parser = argparse.ArgumentParser("MX Intrusion Demo!")
    parser.add_argument('--roi_coordinates', nargs='+', type=int, help='space seperated ROI coordinates in x1y1x2y2 format')
    parser.add_argument("--dfp", default="../../models/YOLO_v8_medium_640_640_3_tflite.dfp", type=str, help="dfp path")
    parser.add_argument("--post_model_path",default="../../models/YOLO_v8_medium_640_640_3_tflite_post.tflite", type=str, help="postprocessing model path")
    parser.add_argument(
        "--input_path", required=True, help="path to video"
    )
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.4, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.6, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--grace_period", type=int, default=50, help="grace period in frames during learning mode")
    return parser

def inRoi(xywh,roi):
    """
    Check where the given rectangle is in the ROI
    """
    x1,y1,w,h = xywh
    x2 = x1+w
    y2 = y1+h
    centroid_box = [(x1+x2)/2,(y1+y2)/2]
    if centroid_box[0]>roi[0] and centroid_box[0]<roi[2] and centroid_box[1]>roi[1] and centroid_box[1]<roi[3]:
        return True
    return False

class IntrusionMxa:
    """
    A demo app to run YOLOv8m on the MemryX MXA.
    """

###################################################################################################
    def __init__(self,args,show):
        """
        The initialization function.
        """

        # Controls
        self.args = args
        self.show = show
        self.dfp = args.dfp
        self.post_model = args.post_model_path
        self.cap_queue = Queue(maxsize=5)
        if "/dev/video" in str(args.input_path):
            self.src_is_cam = True
        else:
            self.src_is_cam = False
        self.vcap = cv2.VideoCapture(args.input_path)
        self.tracker = BYTETracker(args, frame_rate=args.fps)
        self.timer = Timer()
        self.frame_id = 0
        self.dims = (int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # Initialize the model with the stream dimensions
        self.model = YoloModel(stream_img_size=(self.dims[1], self.dims[0], 3))
        self.done = False
        self.color_wheel = np.random.randint(0, 255, (20, 3)).astype(np.int32)
        self.grace_period = args.grace_period  # Grace period from arguments
        
        self.object_store = {}  # Stores all known objects from grace period
        
        self.intruders = {}  # Stores intruder IDs that entered ROI after grace period
        
        if args.roi_coordinates and len(args.roi_coordinates)==4:
            self.roi = args.roi_coordinates
        else:
            # Automatically calculate a centered ROI based on video dimensions
            width, height = self.dims
            self.roi = [int(width * 0.375), int(height * 0.375), 
                       int(width * 0.625), int(height * 0.625)]
        self.detect_counter = 0
        self.roi_color = (0,255,0)
        self.first_detection_printed = False 
        
        # ROI dragging variables
        self.dragging = False
        self.drag_start = None
        self.roi_offset = [0, 0]

###################################################################################################
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for ROI manipulation.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is inside ROI
            if (self.roi[0] <= x <= self.roi[2] and self.roi[1] <= y <= self.roi[3]):
                self.dragging = True
                self.drag_start = (x, y)
                self.roi_offset = [x - self.roi[0], y - self.roi[1]]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                # Calculate new ROI position
                roi_width = self.roi[2] - self.roi[0]
                roi_height = self.roi[3] - self.roi[1]
                
                new_x1 = x - self.roi_offset[0]
                new_y1 = y - self.roi_offset[1]
                
                # Keep ROI within frame boundaries
                new_x1 = max(0, min(new_x1, self.dims[0] - roi_width))
                new_y1 = max(0, min(new_y1, self.dims[1] - roi_height))
                
                self.roi[0] = new_x1
                self.roi[1] = new_y1
                self.roi[2] = new_x1 + roi_width
                self.roi[3] = new_y1 + roi_height
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_start = None

###################################################################################################
    def run(self):
        """
        The function that starts the inference on the MXA.
        """
        accl = AsyncAccl(dfp=self.dfp)
        accl.set_postprocessing_model(self.post_model, model_idx=0)

        # Connect the input and output functions and let the accl run
        accl.connect_input(self.capture_and_preprocess)
        accl.connect_output(self.postprocess)
        accl.wait()

        self.done = True

        cv2.destroyAllWindows()
        self.vcap.release()

###################################################################################################
    def capture_and_preprocess(self):
        """
        Captures a frame for the video device and pre-processes it.
        """
        while True:
            got_frame, frame = self.vcap.read()

            if not got_frame or self.done:
                return None

            if self.src_is_cam and self.cap_queue.full():
                # drop cam frame
                continue
            else:
                # Put the frame in the cap_queue to be processed later
                self.cap_queue.put(frame)

                # Pre-process the frame using the corresponding model
                frame = self.model.preprocess(frame)
                return frame

###################################################################################################
    def postprocess(self, *mxa_output):
        """
        Post-process the MXA output.
        """
        dets = self.model.postprocess(mxa_output)
        # Push the detection results to the queue
        frame = self.cap_queue.get()
        self.cap_queue.task_done()
        if not dets:
            return

        # Draw detection boxes on the frame
        detection_results = []
        for d in dets:
            # print(d["class"])
            x1, y1, w, h = d['bbox']
            x2,y2 = x1+w,y1+h
            result =[x1,y1,x2,y2,d['score']]
            detection_results.append(result)
        detection_results = torch.tensor(detection_results)
        # Filter out detections based on criteria:
        # Box area > 10 pixels (min_box_area)
        # Aspect ratio <= 1.6 (aspect_ratio_thresh)
        # which made it fail to track objects properly
        online_targets = self.tracker.update(detection_results)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            # Add ALL tracked objects without filtering
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
        if self.timer.start_time>0:
            self.timer.toc()
        else:
            self.timer.average_time = 10000

        if self.frame_id < self.grace_period:
            # Store ALL detected object IDs during grace period
            # These objects are considered part of the normal scene "Safe"
            for id in online_ids:
                if id not in self.object_store:
                    self.object_store[id] = True
        else:
            # After grace period ends, check for new objects entering the ROI
            for i,id in enumerate(online_ids):
                if id not in self.object_store and inRoi(online_tlwhs[i],self.roi):
                    print("Intruder detected with id: ",id)
                    self.intruders[id] = True  
                    self.object_store[id] = True  # Also add to known objects to avoid re-detecting

        overlay = frame.copy()
        alpha = 0.5 
        

        intruders_in_roi = False
        if self.frame_id >= self.grace_period:  # Only check after grace period
            for i, id in enumerate(online_ids):
                # Check if this object is a known intruder AND is currently in the ROI
                if id in self.intruders and inRoi(online_tlwhs[i], self.roi):
                    intruders_in_roi = True
                    break
        
       
        if intruders_in_roi:
            self.roi_color = (0, 0, 255)
            roi_width = self.roi[2] - self.roi[0]
            text = "Intrusion Alert!!!!"
            font_scale = 1
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, 2)[0]
    
            if text_size[0] > roi_width:
                font_scale = roi_width / text_size[0]
            text_x = self.roi[0]
            text_y = self.roi[1] - 10
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, font_scale, (0,0,255), 2)
        else:
            self.roi_color = (0,255,0)
        
        cv2.rectangle(overlay, (self.roi[0],self.roi[1]), (self.roi[2],self.roi[3]), self.roi_color, -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        self.timer.tic()
        is_learning = self.frame_id < self.grace_period
        online_im = plot_tracking(
            frame, online_tlwhs, online_ids, frame_id=self.frame_id, fps=1. / self.timer.average_time, is_learning_mode=is_learning
        )
        window_name = "YOLOv8m Tracking"
        cv2.imshow(window_name, online_im)
        
        if self.frame_id == 0:
            cv2.setMouseCallback(window_name, self.mouse_callback)
        
        if cv2.waitKey(20) == ord('q'):
            self.done = True
        
        self.frame_id+=1

###################################################################################################
###################################################################################################
###################################################################################################

def main(args):
    """
    The main funtion
    """
    App = IntrusionMxa(args,show=True)
    App.run()

###################################################################################################

if __name__=="__main__":
    args = make_parser().parse_args()
    main(args)

# eof
