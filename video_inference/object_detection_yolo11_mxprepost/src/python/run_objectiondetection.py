"""
============
Information:
============
Project: YOLOv11 example code on MXA using MXPREPOST    
"""

###############################################################################
# Import necessary libraries ##################################################
###############################################################################
import os
import time
import argparse

import cv2
import numpy as np

from queue import Queue
from threading import Thread

from memryx import mxapi
import mxprepost


###############################################################################
# Class with necessary methods to run the application #########################
###############################################################################

class YoloV11Mxa:
    """
    A demo app to run YOLOv11 on the MemryX MXA.
    """

    ###############################################################################
    # Class constructor ###########################################################
    ###############################################################################

    def __init__(self, video_path, dfp_path, show=True, ):
        """
        The initialization function.
        """

        # Controls
        self.show = show
        self.done = False
        self.dfp_path = dfp_path

        # Stream-related containers
        # CV and Queues
        self.num_frames = 0
        self.cap_queue = Queue(maxsize=4)
        self.dets_queue = Queue(maxsize=5)
        if "/dev/video" in str(video_path):
            self.src_is_cam = True
        else:
            self.src_is_cam = False
        self.vidcap = cv2.VideoCapture(video_path) 
        self.dims = ( int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) )

        # Timing and FPS
        self.dt_index = 0
        self.frame_end_time = 0
        self.fps = 0
        self.dt_array = np.zeros([30])


        # Display and Save Thread
        # Runnting the display and save as a thread enhance the pipeline performance
        # Otherwise, the display_save method can be called from the output method
        self.display_thread = Thread(target=self.display,args=(), daemon=True)

    ###############################################################################
    # Run inference function to init accl and start ###############################
    ###############################################################################

    def run(self):
        """
        The function that starts the inference on the MXA.
        """
        print("dfp path = ", self.dfp_path)
        # accl = AsyncAccl(dfp=self.dfp_path, use_model_shape=(False, False))
        accl = mxapi.MxAccl(
            dfp_path=self.dfp_path,
            local_mode=True,
            use_model_shape=[False, False],
        )
        print("YOLOv11 inference on MX3 started")

        self.display_thread.start()

        start_time = time.time()

        # Initialize mxprepost before connecting streams
        self.prepost = mxprepost.MxPrepost(
            accl=accl,
            task='yolov11-det',
            conf=0.6,
            iou=0.6,
        )

        # Connect the input and output functions and let the accl run
        accl.connect_stream(self.capture_and_preprocess, self.postprocess, stream_id=0, model_id=0)

        accl.start()
        accl.wait()
        self.done = True

        # Join the display thread
        self.display_thread.join()

    ###############################################################################
    # Input and Output functions ##################################################
    ###############################################################################
    # Capture frames for streams and pre process
    def capture_and_preprocess(self, stream_id=0):
        """
        Captures a frame for the video device and pre-processes it.
        """

        while True:

            got_frame, frame = self.vidcap.read()

            if not got_frame:
                return None

            if self.src_is_cam and self.cap_queue.full():
                # drop the frame and try again
                continue
            else:
                self.num_frames += 1
                
                # Put the frame in the cap_queue to be overlayed later
                self.cap_queue.put(frame)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Preprocess frame using mxprepost
                rgb_frame = self.prepost.preprocess(rgb_frame)
                return rgb_frame
        
    ###############################################################################
    # Post process the output from MXA
    def postprocess(self, mxa_output, stream_id=0):
        """
        Post-process the MXA output.
        """

        # Post-process the MXA output
        dets = self.prepost.postprocess(mxa_output, self.dims[1], self.dims[0])

        # Push the results to the queue to be used by the display_save thread
        self.dets_queue.put(dets)

        # Calculate current FPS
        self.dt_array[self.dt_index] = time.time() - self.frame_end_time
        self.dt_index +=1
        
        if self.dt_index % 15 == 0:
            self.fps = 1 / np.average(self.dt_array)

            if self.dt_index >= 30:
                self.dt_index = 0
        
        self.frame_end_time = time.time()

    ###############################################################################
    # Display the output and show if opted in
    def display(self):
        """
        Continuously draws boxes over the original image for each stream and displays them in separate windows.
        """
        while not self.done:
             
            # Get the frame from and the dets from the relevant queues
            frame = self.cap_queue.get()
            result = self.dets_queue.get()
            self.cap_queue.task_done()
            self.dets_queue.task_done()

            # Draw detection boxes using mxprepost draw method
            frame = self.prepost.draw(frame, result)

            if self.fps > 1:
               txt = f"{self.fps:.1f} FPS"
            else:
               txt = f""
            frame = cv2.putText(frame, txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2) 

            # Show the frame
            if self.show:

                cv2.imshow('YOLOv11 on MemryX MXA', frame)

                # Exit on a key press
                if cv2.waitKey(1) == ord('q'):
                    self.done = True
                    cv2.destroyAllWindows()
                    self.vidcap.release()
                    exit(1)

###############################################################################
# Main function of the application ############################################
###############################################################################

def main(args):
    """
    The main function
    """

    yolo11_inf = YoloV11Mxa(video_path = args.video_path, dfp_path=args.dfp, show=args.show,)
    yolo11_inf.run()

###############################################################################

if __name__=="__main__":
 
    # The args parser to parse input paths
    parser = argparse.ArgumentParser(description="\033[34mRun MX3 real-time inference with options for DFP file and post model file path.\033[0m")
    
    parser.add_argument('-d', '--dfp', 
                        type=str, 
                        default="../../models/YOLO11_small_640_640_3_onnx.dfp", 
                        help="Specify the path to the compiled DFP file. Default is '../../models/YOLO11_small_640_640_3_onnx.dfp'.")

    parser.add_argument('--video_path',  dest="video_path", 
                        action="store", 
                        default='/dev/video0',
                        help="the path to video file to run inference on. Use '/dev/video0' for a webcam \n (Default: '/dev/video0')")

    parser.add_argument('--no_display', dest="show", 
                        action="store_false", 
                        default=True,
                        help="Optionally turn off the video display")

    args = parser.parse_args()

    # Call the main function
    main(args)