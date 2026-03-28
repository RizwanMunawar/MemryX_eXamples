"""
Copyright (c) 2019-2026 MemryX Inc.
All Rights Reserved.

============
File Name:      runGame.py
Project:        fun_examples/handPong_game
============


============
Description:
classic game of pong using computer vision to detect hand movements and 
link it to the games player paddles. Using the mediapipe model, it can 
detect and diffrentiate between left hand and right hand, and assigns 
player 1 and player 2 to it, respectevly. The game also keeps players 
scores and display it on the screen.
============


======
Notes:

======



======== 
Authors:
Raid Al-Tamimi - Creator
 
  #
 ###
#####
  ☺
 /|\
 / \

========

======== 
Version history:
V0.9.0_alpha - 19/nov/2025 - Raid - Created Game
V0.9.0_beta - 1/jan/2026 - Raid - Taking the game from alpha stage to beta testing
V0.9.1_beta - 13/jan/2026 - Raid - Added a system force terminate to close program and all threads once "X" is pressed
V1.0.0        - 20/jan/2026 - Raid - Taking the game from Beta stage to full release
========

======== 
Latest tested dependencies:
pygame          v2.6.1
python          v3.12.3
mx_runtime      v2.2.0.dev13
opencv-python   v4.11.0.86
numpy           v1.26.4
========

"""
import pygame
import os
from pygame.locals import *
import threading, json, cv2, sys, pickle
import numpy as np
from MxHandPose import MxHandPose
import argparse
import threading


pygame.init()

WIDTH, HEIGHT = 1280, 720
WIN = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("MemryX - Pong")

prog_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(prog_dir, '../../assets', 'mx3.png')
image = pygame.image.load(image_path).convert_alpha()
img_rect=image.get_rect()


FPS = 60
WHITE = (255,255,255)
BLACK = (0,0,0)
RED   = (255,0,0)

PADDLE_HEIGHT, PADDLE_WIDTH = 200,30
BALL_SIZE = 54

clk = pygame.time.Clock()

fps_font = pygame.font.SysFont("Arial", 30)
scores_font = pygame.font.SysFont("Arial", 50)

debugging=1



###############################################################################
# Parse command-line arguments for debugging mode (-D) and DFP file (-d)
###############################################################################
parser = argparse.ArgumentParser(description="Run MX3 real-time inference with options for DFP file and Debugging mode")
parser.add_argument('-d', '--dfp', type=str, default="../../models/models.dfp", help="Specify the path to the compiled DFP file. Default is '../../models/models.dfp'.")
parser.add_argument('-D', '--debug',action='store_true',help='Run the program in debugging mode')
args = parser.parse_args()

dfp_path = args.dfp
debugging = args.debug



###############################################################################
# Check if the models folder is created in the correct path and creates it if not
###############################################################################
if not os.path.exists("../../models"):
    print("WARNING: models folder does not exist in the correct directory. creating folder...")
    os.system("mkdir -p ../../models")


###############################################################################
# Check if the dfp file exists or not
###############################################################################
if os.path.abspath(dfp_path) and os.path.isfile(dfp_path):
    print("\033[93mCompiled DFP file found at {}.\033[0m".format(dfp_path))
else:
    print("\033[93mError: DFP file not found.\033[0m")
    exit()




def video_capture():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    camera.set(cv2.CAP_PROP_FPS, 60)
        
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)


    return camera


def camera_read(mxpose,cap):

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        

    if mxpose.full():
        pass
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mxpose.put(frame)

        

def get_frame(mxposes,hand_pos):
    if not mxposes.empty():
        annotated_frame = mxposes.get()

        frame = draw_on_frame(annotated_frame,hand_pos)

        frame = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], 'RGB')
        return frame
    else:
        return "empty"
        
def draw_on_frame(annotated_frame,hand_pos):

    handkeypoints_lst, handtype_lst = get_handkeypoints_handtype(annotated_frame)
    img = annotated_frame.image
    if debugging:
        print(handkeypoints_lst)
    for idx,handtype in enumerate(handtype_lst):
        img = drawLandmarks(img, [handkeypoints_lst[idx]], (handtype == "Left"), hand_pos)

    return img

def get_handkeypoints_handtype( annotated_frame):
    handkeypoints_lst = []
    handtype_lst      = []
    for handpose in annotated_frame.handposes:
        hp_reshaped = handpose.landmarks.reshape(21, 3).astype(np.int32)
        singlehand_keypoints = [(int(x), int(y)) for x,y,z in hp_reshaped]
        handkeypoints_lst.append(singlehand_keypoints)
        handtype_lst.append(handpose.handedness)

    return handkeypoints_lst, handtype_lst

def drawLandmarks(frame, data, is_left, hand_pos):
    allhands=data
    color_red=(255,0,0)
    color_green=(0,255,0)
    color_blue=(0,0,255)

    if is_left:
        color = (255,0,255)
        
    else:
        color = (0,255,255)
    for myHand in allhands:
           
        cv2.line(frame,(myHand[5][0],myHand[5][1]),(myHand[9][0],myHand[9][1]),color_red,2) 
        cv2.line(frame,(myHand[9][0],myHand[9][1]),(myHand[13][0],myHand[13][1]),color_green,2) 
        cv2.line(frame,(myHand[13][0],myHand[13][1]),(myHand[17][0],myHand[17][1]),color_blue,2)
         
        if is_left:
            hand_pos.set_pos("left",((myHand[5][1]+myHand[9][1]) // 2))
        else:
            hand_pos.set_pos("right",((myHand[5][1]+myHand[9][1]) // 2))
        for i in myHand:
            cv2.circle(frame,(i[0],i[1]),4,(23,90,10),1)
        for i in myHand:
            cv2.circle(frame,(i[0],i[1]),3,(0,0,255),-1)
    return frame
    


class Hand_positions:
    left_pos=0
    right_pos=0

    def set_pos(self,hand,pos):
        if hand == "right":
            self.right_pos=pos
        if hand == "left":
            self.left_pos=pos

    def get_pos(self,hand):
        if hand == "right":
            return self.right_pos
        if hand == "left":
            return self.left_pos



class Paddle:

    COLOR = BLACK
    VEL = 11

    def __init__( self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height= height

    def draw(self,win):
        pygame.draw.rect(win,self.COLOR, (self.x,self.y,self.width,self.height))
    
    def move(self, up=True):
        if up:
            self.y -= self.VEL
            if self.y <= 0:
                self.y = 0
        else:
            self.y += self.VEL
            if self.y + self.height>= HEIGHT:
                self.y = HEIGHT - self.height
        return self.y
    def move_to_pos(self,pos):
        self.y = pos
    


class Ball:
    MAX_VEL = 7
    COLOR = BLACK

    def __init__(self,x,y,size):
        self.x = x
        self.y = y
        self.size = size
        self.x_vel = self.MAX_VEL
        self.y_vel = 0

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel



    def draw(self,win):
        pygame.draw.rect(win,self.COLOR,(self.x-self.size//2,self.y-self.size//2, self.size,self.size))
        img_rect.topleft = (self.x-self.size//2,self.y-self.size//2)
        win.blit(image,img_rect)
        

class scores:

    COLOR = RED
    def __init__(self,p1,p2):
        self.p1 = p1
        self.p2 = p2

    def point(self,player):
        if player == 1:
            self.p1 +=1
        if player == 2:
            self.p2 +=1

    def draw(self,win):
        text_surface_p1 = scores_font.render("{}".format(self.p1), True, self.COLOR)
        text_surface_p2 = scores_font.render("{}".format(self.p2), True, self.COLOR)

        text_rect_p1 = text_surface_p1.get_rect()
        text_rect_p2 = text_surface_p2.get_rect()

        text_rect_p1.center = (WIDTH*0.25, 30) 
        text_rect_p2.center = (WIDTH*0.75, 30) 

        win.blit(text_surface_p1, text_rect_p1)
        win.blit(text_surface_p2, text_rect_p2)


def draw(win, paddles, ball, scores, mx_poses,hand_pos):
    win.fill(WHITE)
     

    cam_img = get_frame(mx_poses,hand_pos)
    if cam_img != "empty":
        win.blit(cam_img, (0, 0))
    else:
        if debugging:
            print("feed is empty, ignore...")
    
    
    for paddle in paddles:
        paddle.draw(win)


        scores.draw(win)

    ball.draw(win)  
    if debugging:
        fps_text = fps_font.render(f"FPS: {int(clk.get_fps())}", True, (255,0,255))
        win.blit(fps_text, (10, 10))  
    pygame.display.update()


def handle_paddle_movement(left_hand_pos,right_hand_pos,left_paddle,right_paddle):

    right_paddle.move_to_pos(right_hand_pos)
    left_paddle.move_to_pos(left_hand_pos)



def handle_collision(ball, left_paddle,right_paddle):
    VEL_INCREMENTS=PADDLE_HEIGHT//5
    
    if ball.y <= 0 or ball.y + ball.size >=HEIGHT:
        ball.y_vel *= -1
    

    if ball.x_vel < 0:
        if ball.y >= left_paddle.y and ball.y <= left_paddle.y + left_paddle.height:
            if ball.x - ball.size//2<= left_paddle.x+left_paddle.width: 
                if ball.y >= left_paddle.y and ball.y < left_paddle.y+VEL_INCREMENTS*1:
                    ball.x_vel =20
                    ball.y_vel =-16
                elif ball.y >= left_paddle.y+VEL_INCREMENTS*1 and  ball.y < left_paddle.y+VEL_INCREMENTS*2:
                    ball.x_vel =16
                    ball.y_vel =-12
                elif ball.y >= left_paddle.y+VEL_INCREMENTS*2 and  ball.y < left_paddle.y+VEL_INCREMENTS*3:
                    ball.x_vel =15
                    ball.y_vel =0
                elif ball.y >= left_paddle.y+VEL_INCREMENTS*3 and  ball.y < left_paddle.y+VEL_INCREMENTS*4:
                    ball.x_vel =16
                    ball.y_vel =12
                elif ball.y >= left_paddle.y+VEL_INCREMENTS*4 and  ball.y < left_paddle.y+VEL_INCREMENTS*5:
                    ball.x_vel =20
                    ball.y_vel =16
                      
        if ball.x < left_paddle.x+left_paddle.width//2:
            return 2


    else:
        if ball.y >= right_paddle.y and ball.y <= right_paddle.y + right_paddle.height:  
            if ball.x + ball.size//2 >= right_paddle.x: 
                if ball.y >= right_paddle.y and  ball.y < right_paddle.y+VEL_INCREMENTS*1:
                    ball.x_vel =-20
                    ball.y_vel =-16
                elif ball.y >= right_paddle.y+VEL_INCREMENTS*1 and  ball.y < right_paddle.y+VEL_INCREMENTS*2:
                    ball.x_vel =-16
                    ball.y_vel =-12
                elif ball.y >= right_paddle.y+VEL_INCREMENTS*2 and  ball.y < right_paddle.y+VEL_INCREMENTS*3:
                    ball.x_vel =-15
                    ball.y_vel =0
                elif ball.y >= right_paddle.y+VEL_INCREMENTS*3 and  ball.y < right_paddle.y+VEL_INCREMENTS*4:
                    ball.x_vel =-16
                    ball.y_vel =12
                elif ball.y >= right_paddle.y+VEL_INCREMENTS*4 and  ball.y < right_paddle.y+VEL_INCREMENTS*5:
                    ball.x_vel =-20
                    ball.y_vel =16
        if ball.x > right_paddle.x+right_paddle.width//2:
            return 1
          
    
    ball.move()


def main():

    mx_pose        = MxHandPose(dfp_path, num_hands=2)

    cap = video_capture()


    
    run = True

    left_paddle = Paddle(10,HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
    right_paddle = Paddle(WIDTH - 10 - PADDLE_WIDTH,HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)

    hand_positions = Hand_positions()

    ball = Ball(WIDTH//2,HEIGHT//2,BALL_SIZE)
    p_score = scores(0,0)
    while run:
        camera_read(mx_pose,cap)
        clk.tick(FPS)
        draw(WIN, [left_paddle,right_paddle], ball, p_score,mx_pose,hand_positions)
        if debugging:
            print("---LEFT HAND POS IS:%d " %(hand_positions.get_pos("left")))
            print("---RIGHT HAND POS IS:%d " %(hand_positions.get_pos("right")))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        if not run:
            break
        
        key = pygame.key.get_pressed()


        handle_paddle_movement(hand_positions.get_pos("left"),hand_positions.get_pos("right"), left_paddle,right_paddle)
        check_point = handle_collision(ball,left_paddle,right_paddle)
        if check_point == 1:
            ball.x = WIDTH//2
            ball.y = HEIGHT//2
            ball.y_vel=0
            ball.x_vel=-4
            p_score.point(1)
        if check_point == 2:
            ball.x = WIDTH//2
            ball.y = HEIGHT//2
            ball.y_vel=0
            ball.x_vel=4
            p_score.point(2)

    pygame.quit()
    mx_pose.stop()
    cap.release()
    os._exit(0)

if __name__ == '__main__':
    main()
