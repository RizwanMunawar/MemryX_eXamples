import os
import argparse
import random
import shutil
from pathlib import Path
import numpy as np

import kagglehub
from PIL import Image
import matplotlib.pyplot as plt

from MXFace import MXFace, AnnotatedFrame

def load_images():
    dataset_dir = Path("../assets/Friends")
    train_dir = dataset_dir / "Train"
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print("Downloading sample dataset...")
        downloaded_path = kagglehub.dataset_download("amiralikalbasi/images-of-friends-character-for-face-recognition")
        downloaded_path = Path(downloaded_path) / 'Friends'  # Convert to Path object for convenience
        shutil.copytree(str(downloaded_path), dataset_dir)
        print(f"Dataset downloaded to {dataset_dir}")
    
    # Get the list of child directories in the training directory
    child_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # Select two random child directories
    selected_dirs = [random.choice(child_dirs), random.choice(child_dirs)]
    
    # Select one random image from each directory
    images = []
    for dir_name in selected_dirs:
        dir_path = os.path.join(train_dir, dir_name)
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        if not image_files:
            raise ValueError(f"No images found in directory: {dir_path}")
        
        # Choose a random image and load it
        random_image = random.choice(image_files)
        image_path = os.path.join(dir_path, random_image)
        images.append(Image.open(image_path).convert('RGB'))
    
    return images[0], images[1]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process two image paths.")
    parser.add_argument(
        '--image_paths', 
        type=str, 
        nargs=2, 
        default=None, 
        help="Paths to the two images, separated by a space"
    )
    parser.add_argument(
        '--model_dir', 
        type=str, 
        default='../models', 
        help="Path "
    )
    return parser.parse_args()

def plot_images(annotated_frame_1: AnnotatedFrame, annotated_frame_2: AnnotatedFrame, similarity: float, verified: bool):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.01)
    
    for idx, (frame, ax) in enumerate(zip([annotated_frame_1, annotated_frame_2], axes)):
        # Draw each image
        ax.imshow(frame.image)
        ax.axis("off")
        ax.set_title(f"Image {idx + 1}")
        
        # Draw bounding boxes and keypoints for each detected face
        for face in frame.detected_faces:
            left, top, width, height = face.bbox
            rect = plt.Rectangle((left, top), width, height, linewidth=2, edgecolor="cyan", facecolor="none")
            ax.add_patch(rect)
                
            for (kx, ky) in face.keypoints:
                ax.plot(kx, ky, "ro", markersize=5)

    # Display similarity and verification status between images
    plt.figtext(0.5, 0.92, f"Similarity: {similarity:.2f}\nVerified: {'Yes' if verified else 'No'}", ha="center", fontsize=12, weight="bold")
    plt.gcf().set_size_inches(12, 6)
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    if args.image_paths:
        image1 = Image.open(args.image_paths[0]).convert('RGB')
        image2 = Image.open(args.image_paths[1]).convert('RGB')
    else:
        image1, image2 = load_images()

    mx_face = MXFace(args.model_dir)

    mx_face.put(np.array(image1))
    mx_face.put(np.array(image2))

    annotated_frame_1 = mx_face.get()
    annotated_frame_2 = mx_face.get()

    if annotated_frame_1.num_detections:
        face_embedding_1 = annotated_frame_1.detected_faces[0].embedding
    else:
        raise ValueError("No face detected in image 1!")

    if annotated_frame_2.num_detections:
        face_embedding_2 = annotated_frame_2.detected_faces[0].embedding
    else:
        raise ValueError("No face detected in image 2!")

    similarity = MXFace.cosine_similarity(face_embedding_1, face_embedding_2)
    verified = similarity >= MXFace.cosine_threshold

    plot_images(annotated_frame_1, annotated_frame_2, similarity, verified)

    mx_face.stop()

