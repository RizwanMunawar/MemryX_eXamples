import os
import argparse
from queue import Queue, Empty
import time
from collections import namedtuple

import cv2 as cv
import numpy as np

from memryx import AsyncAccl, SchedulerOptions, ClientOptions
from FaceDetector import FaceDetector
import signal
from threading import Event

stop_flag = Event()

# Function to handle Ctrl+C (SIGINT)
def signal_handler(sig, frame):
    print("Exiting program with Ctrl+C...")
    stop_flag.set()  # Signal threads to stop

class FaceCropCartoonizer:
    def __init__(self, cam, stop_flag, src_is_cam, show=True):
        self.cam = cam
        self.show = show
        self.src_is_cam = src_is_cam

        # Create a namedtuple for shape
        Shape = namedtuple("Shape", ["height", "width"])
        self.face_model = FaceDetector(model_input_shape=Shape(height=128, width=128))

        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))

        # init queues
        self.frame_queue = Queue(maxsize=20)
        self.relay_frame_queue = Queue()
        self.face_dets_queue = Queue()

        self.stop_flag = stop_flag
        

    def get_face_box(self, det):
        ymin = int(det[0] * self.input_height)
        xmin = int(det[1] * self.input_width)
        ymax = int(det[2] * self.input_height)
        xmax = int(det[3] * self.input_width)

        # Clamp to image boundaries
        xmin = max(0, min(xmin, self.input_width - 1))
        xmax = max(0, min(xmax, self.input_width - 1))
        ymin = max(0, min(ymin, self.input_height - 1))
        ymax = max(0, min(ymax, self.input_height - 1))

        # Ensure valid box: xmin < xmax, ymin < ymax
        if xmax <= xmin or ymax <= ymin:
            return None

        return [xmin, ymin, xmax, ymax]

    def get_frame_face(self):

        while True:

            if self.stop_flag.is_set():
                print("Stop flag set, exiting get_frame_face loop.")
                return None

            if not self.src_is_cam:
                time.sleep(0.03)  # regulate speed for video input
            
            ok, frame = self.cam.read()  # Read frame from camera

            if not ok:
                print("EOF")  # Handle end of stream
                return None
                
            if self.frame_queue.full():
                # drop frame (and ONLY in this first model!)
                # dependent models shouldn't drop anything
                continue
            else:
                self.frame_queue.put(frame)  # Put the original frame in the queue
                out = self.face_model.preprocess(frame)
                out = np.expand_dims(out, 0)
                return out

    def process_face_output(self, *ofmaps):
        if len(ofmaps) == 1:
            ofmaps = ofmaps[0]

        # Process and store the face detection result
        dets = self.face_model.postprocess(*ofmaps)

        self.face_dets_queue.put(dets)


    def preprocess_cartoonizer(self, img):
        arr = np.array(cv.resize(img, (512, 512))).astype(np.float32)
        arr = arr / 127.5 - 1

        # add batch dimension and change to channel first format
        # Ensure that the input shape to the accelerator matches the input shapes of the ONNX model.
        arr = np.expand_dims(arr, 0)
        arr = np.transpose(arr, (0, 3, 1, 2))
        return arr

    def get_frame_cartoonizer(self):
        
        while not self.stop_flag.is_set() or not self.frame_queue.empty():

            try:
                frame = self.frame_queue.get(timeout=0.1)  # Wait for a frame to be available
            except Empty:
                continue # No frame available, check stop flag and loop again
            
            self.relay_frame_queue.put(frame)
            return self.preprocess_cartoonizer(frame)
        
        print("Stop flag set and frame queue empty, exiting get_frame_cartoonizer loop.")
        return None

    def postprocess_cartoonizer(self, frame, original_shape):

        # get rid of batch dim and change to channel last format
        # note that the output from the accelerator matches the output shapes from the ONNX model.
        # i.e. channel last format with batch dim
        frame = np.squeeze(frame, 0)
        frame = np.transpose(frame, (1, 2, 0))
        frame = (frame + 1) * 127.5
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = cv.resize(frame, original_shape)
        return frame

    def process_cartoonizer_output(self, *ofmaps):

        dets = self.face_dets_queue.get()
        display_img = self.relay_frame_queue.get()

        face_cnt = dets.size
        if face_cnt > 0:
            # Only postprocess cartoonized image if there are detected faces
            cartoonized_img = self.postprocess_cartoonizer(
                ofmaps[0], (self.input_width, self.input_height)
            )

            # Process each detected face
            for det in dets:
                # Get coordinates for the detected face
                coord = self.get_face_box(det)
                if coord is None:
                    continue

                xmin, ymin, xmax, ymax = coord

                # Extract cartoonized patch
                cartoon_patch = cartoonized_img[ymin:ymax, xmin:xmax]

                # Convert cartoonized face patch to grayscale like a ghost!
                gray_cartoon_patch = cv.cvtColor(cartoon_patch, cv.COLOR_RGB2GRAY)

                # Convert grayscale patch to 3-channel
                gray_3ch = cv.cvtColor(gray_cartoon_patch, cv.COLOR_GRAY2RGB)

                # Replace region in display_img with cartoonized patch
                display_img[ymin:ymax, xmin:xmax] = gray_3ch

                # Draw bounding boxes around detected faces
                display_img = cv.rectangle(
                    display_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3
                )

        # display
        if self.show:
            self.display(display_img)

    def display(self, img):
        cv.imshow("display", img)  # Show the image in a window
        if cv.waitKey(1) == ord("q"):  # Exit on 'q' key press
            print("Exit signal received. Closing...")
            stop_flag.set()  # Signal threads to stop

            
# Main function to run the application with MemryX AsyncAccl
def run_dfps(args):

    face_sche_opts = SchedulerOptions()
    face_sche_opts.frame_limit = args.frame_limit
    
    cart_sche_opts = SchedulerOptions()
    cart_sche_opts.frame_limit = args.frame_limit
    
    client_opts = ClientOptions(True, 30.0) # smooth result to 30 FPS

    # Initialize AsyncAccl for face detection and cartoonizer
    accl_face = AsyncAccl(args.dfp_face, scheduler_options=face_sche_opts, client_options=client_opts)
    accl_cartoonizer = AsyncAccl(args.dfp_cartoonizer, scheduler_options=cart_sche_opts, client_options=client_opts)

    # Connect face detection input and output
    accl_face.connect_input(app.get_frame_face)
    accl_face.connect_output(app.process_face_output)

    # Connect cartoonizer input and output
    accl_cartoonizer.connect_input(app.get_frame_cartoonizer)
    accl_cartoonizer.connect_output(app.process_cartoonizer_output)

    # Wait for the asynchronous processing
    accl_face.wait()
    accl_cartoonizer.wait()  # Wait for the asynchronous processing


if __name__ == "__main__":

    # Handle Ctrl+C signal
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Face Crop & Conditional Cartoonizer")
    parser.add_argument(
        "-df",
        "--dfp_face",
        type=str,
        default="../../models/face_det.dfp",
        help="Path to the Face Detection DFP file (default: ../../models/face_det.dfp)",
    )
    parser.add_argument(
        "-dc",
        "--dfp_cartoonizer",
        type=str,
        default="../../models/cartoonizer.dfp",
        help="Path to the Cartoonizer DFP file (default: ../../models/cartoonizer.dfp)",
    )
    parser.add_argument(
        "--no_display",
        dest="show",
        action="store_false",
        default=True,
        help="Turn off video display",
    )
    
    parser.add_argument(
        "--frame_limit",
        "-f",
        type=int,
        default=3,
        help="Number of frames to process before swapping out to another DFP (default: 3)",
    )
    
    parser.add_argument("--video", type=str, help="Path to video file (instead of cam)")

    args = parser.parse_args()

    # Set up input source
    if args.video:
        input_source = args.video
    else:
        input_source = 0 # default cam

    cam = cv.VideoCapture(input_source)

    # Initialize the application
    app = FaceCropCartoonizer(cam, stop_flag, src_is_cam=args.video, show=args.show)

    # Run the application with the provided DFP
    run_dfps(args)

    # After shutdown, release the camera and destroy all windows
    cam.release()
    cv.destroyAllWindows()
