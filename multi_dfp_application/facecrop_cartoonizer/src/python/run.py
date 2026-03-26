import os
import argparse
from queue import Queue
import time
from collections import namedtuple

import cv2 as cv
import numpy as np

from memryx import AsyncAccl, SchedulerOptions, ClientOptions
from FaceDetector import FaceDetector
import signal
import sys

run_flag = False


# Function to handle Ctrl+C (SIGINT)
def signal_handler(sig, frame):
    print("Exiting program with Ctrl+C...")
    cleanup_and_exit()


# Cleanup and exit function to properly terminate resources
def cleanup_and_exit():
    """
    Releases all resources, closes the window, and exits the program.
    """
    global run_flag
    run_flag = False
    cv.destroyAllWindows()  # Close any OpenCV windows

    print("Resources released. Exiting.")
    sys.exit(0)


class App:
    def __init__(self, cam, scale=1.0):
        self.cam = cam

        # Create a namedtuple for shape
        Shape = namedtuple("Shape", ["height", "width"])
        self.face_model = FaceDetector(model_input_shape=Shape(height=128, width=128))

        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT) * scale)
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH) * scale)

        # init queues
        self.frame_queue = Queue()
        self.relay_frame_queue = Queue()
        self.face_dets_queue = Queue()

        self.face_frame_cnt = 0  # Counter for face frames processed
        self.cartoonizer_frame_cnt = 0  # Counter for cartoonizer frames processed

        global run_flag
        run_flag = True

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

        global run_flag
        if run_flag is False:
            return None

        while True:
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

        # self.face_frame_cnt += 1  # Increment face frame count
        # print("face_frame_cnt:", self.face_frame_cnt, flush=True)

    def preprocess_cartoonizer(self, img):
        arr = np.array(cv.resize(img, (512, 512))).astype(np.float32)
        arr = arr / 127.5 - 1

        # add batch dimension and change to channel first format
        # Ensure that the input shape to the accelerator matches the input shapes of the ONNX model.
        arr = np.expand_dims(arr, 0)
        arr = np.transpose(arr, (0, 3, 1, 2))
        return arr

    def get_frame_cartoonizer(self):

        global run_flag
        if run_flag is False:
            return None

        frame = self.frame_queue.get()
        self.relay_frame_queue.put(frame)
        return self.preprocess_cartoonizer(frame)

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
                gray_cartoon_patch = cv.cvtColor(cartoon_patch, cv.COLOR_BGR2GRAY)

                # Convert grayscale patch to 3-channel
                gray_3ch = cv.cvtColor(gray_cartoon_patch, cv.COLOR_GRAY2BGR)

                # Replace region in display_img with cartoonized patch
                display_img[ymin:ymax, xmin:xmax] = gray_3ch

                # Draw bounding boxes around detected faces
                display_img = cv.rectangle(
                    display_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3
                )

        self.display(display_img)

        # self.cartoonizer_frame_cnt += 1  # Increment cartoonizer frame count
        # print("cartoonizer_frame_cnt:", self.cartoonizer_frame_cnt, flush=True)

    def display(self, img):
        cv.imshow("display", img)  # Show the image in a window
        time.sleep(0.01)  # Sleep to allow the window to update
        if cv.waitKey(1) == ord("q"):  # Exit on 'q' key press
            self.cam.release()  # Release the camera resource
            cv.destroyAllWindows()  # Close OpenCV windows
            os._exit(0)


# Main function to run the application with MemryX AsyncAccl
def run_dfps(dfp_face, dfp_cartoon):

    face_sche_opts = SchedulerOptions(
              frame_limit=20,
              time_limit=0,
              ifmap_queue_size=20,
              ofmap_queue_size=36
            )
    cart_sche_opts = SchedulerOptions(
                frame_limit=0,
                time_limit=50,
                ifmap_queue_size=40,
                ofmap_queue_size=40
            )
    client_opts = ClientOptions(True, 30.0) # smooth result to 30 FPS

    # Initialize AsyncAccl for face detection and cartoonizer
    accl_face = AsyncAccl(dfp_face, scheduler_options=face_sche_opts, client_options=client_opts)
    accl_cartoonizer = AsyncAccl(dfp_cartoon, scheduler_options=cart_sche_opts, client_options=client_opts)

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
    parser.add_argument("--video", type=str, help="Path to video file (instead of cam)")

    args = parser.parse_args()

    # Set up input source
    if args.video:
        input_source = args.video
    else:
        input_source = 0 # default cam

    cam = cv.VideoCapture(input_source)

    # Initialize the application
    app = App(cam)

    # Run the application with the provided DFP
    run_dfps(args.dfp_face, args.dfp_cartoonizer)

    # After shutdown, release the camera and destroy all windows
    cam.release()
    cv.destroyAllWindows()
