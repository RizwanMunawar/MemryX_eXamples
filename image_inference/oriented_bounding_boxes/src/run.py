import os
import urllib.request
import zipfile
import argparse
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics.utils import ops
from ultralytics.data.augment import LetterBox
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from MXObb import MXObb

def draw_boxes(annotated_frame, image_path):
    fig, ax = plt.subplots(1, figsize=(12.8, 7.2))
    ax.imshow(annotated_frame.image)
    
    # Minimize white space around the image
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Assign colors to classes
    colors = plt.get_cmap('tab10')
    class_colors = {}
    
    # Draw each bounding box
    for detection in annotated_frame.detections:
        cls = detection.cls
        if cls not in class_colors:
            class_colors[cls] = colors(len(class_colors))
        
        x, y, width, height = detection.bbox
        x1 = x - width / 2
        y1 = y - height / 2
        rect = patches.Rectangle(
            (x1, y1), 
            width, 
            height, 
            linewidth=2, 
            edgecolor=class_colors[cls],
            facecolor="none", 
            angle=np.degrees(detection.rot), 
            rotation_point='center',
        )
        ax.add_patch(rect)
    
    # Add legend for detected classes
    handles = [patches.Patch(color=color, label=cls) for cls, color in class_colors.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    
    # Show the plot
    plt.axis("off")
    plt.show()

def check_models_dir(model_dir):
    dfp_path = f'{model_dir}/yolov8s-obb.dfp'
    post_path = f'{model_dir}/yolov8s-obb_post.onnx'

    if args.models_dir == '../models': 
        if not all(os.path.exists(f) for f in [dfp_path, post_path]):
            os.system('cd ..; chmod +x models/download_model.sh; ./models/download_model.sh')
    else:
        if not os.path.exists(dfp_path) or not os.path.exists(dfp_path):
            raise ValueError(f'{args.models_dir} is invalid.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run object detection on an image.")
    parser.add_argument('--image_path', type=str, default='../assets/parking_lot.jpg', help="Path to the image file.")
    parser.add_argument('--models_dir', type=str, default='../models', help="Path to the directory containting the .dfp.")
    args = parser.parse_args()

    check_models_dir(args.models_dir)

    # Load yolo OBB on the MXA
    mx_obb = MXObb(args.models_dir)

    # Load the image from the specified path
    image = Image.open(args.image_path).convert('RGB')
    image = np.array(image)
    
    mx_obb.put(image)
    annotated_frame = mx_obb.get()

    #draw_boxes(annotated_frame.image, annotated_frame.detections[:, :5])
    draw_boxes(annotated_frame, args.image_path)

    mx_obb.stop()

