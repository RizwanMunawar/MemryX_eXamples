import heapq
import cv2 as cv
import os

class Displayer:
    def __init__(self):
        self.min_heap_ = []
        self.curr_frame_cnt_ = 0
        
    def push(self, frame_cnt, frame):
        heapq.heappush(self.min_heap_, (frame_cnt, frame))
        
        while len(self.min_heap_) > 0:
            frame_cnt, curr_frame = heapq.heappop(self.min_heap_)

            if frame_cnt != self.curr_frame_cnt_:
                heapq.heappush(self.min_heap_, (frame_cnt, curr_frame))
                break

            print ("Displaying frame:", self.curr_frame_cnt_)
            
            self.curr_frame_cnt_ += 1
            
            # Display the frame
            cv.imshow('Emotions', curr_frame)  # Show the image in a window
            if cv.waitKey(1) == ord('q'):  # Exit on 'q' key press
                self.cam.release()  # Release the camera resource
                cv.destroyAllWindows()  # Close OpenCV windows
                os._exit(0)
