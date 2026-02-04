# YOLO Object Detection with Hardware-Accelerated Video Decoding

This example demonstrates real-time object detection on multiple input streams using hardware-accelerated video decoding on supported platforms via FFmpeg. The application supports 20 variations of YOLO models on MemryX accelerators. Hardware acceleration via FFmpeg is available for H.264, H.265, and MJPEG-encoded video and camera inputs across Intel, AMD, RockChip, and Broadcom platforms. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align="center">
  <img src="YOLO_preview.gif" alt="Object Detection Example" width="45%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Yolov8, Yolov9, Yolov10, Yolov11](https://developer.memryx.com/model_explorer/models.html)                                            |
| **Model Type**       | Object Detection                                                      |
| **Framework**        | [ONNX](https://onnx.ai/)                                                   |
| **Model Source**     | [Referennce](https://developer.memryx.com/model_explorer/models.html) |
| **Pre-compiled DFP** | Use provided script to download                                           |
| **Dataset**          | [COCO](https://docs.ultralytics.com/datasets/detect/coco/) |
| **Model Resolution**            | 320x320, 480x480, 640x640, 800x800                                                    |
| **Output**           | Bounding box coordinates with object probabilities |
| **OS**           | Linux |
| **License**          | [GPL](LICENSE.md)                                        |

## Requirements

Before running the application, ensure that FFmpeg is installed:

```bash
# FFmpeg installation
sudo apt install ffmpeg
```
## Running the Application

### Step 1: Download Pre-compiled DFPs

Because this example supports many models, the pre-compiled DFPs can be downloaded and unzipped into the expected directories using the provided bash script. The script can be used
to download all 20 supported models or a subset of said models depending on your preference. If you're unsure, we recommend downloading all models using the `--all` argument:

```bash
cd yolo_object_detection_hw_decoding/

# Run to download all 20 supported models:
./download_models.sh --all

# Run to download a single model (valid options listed in Step 2): 
./download_models.sh 8s640

# Run to download multiple models (valid options listed in Step 2):
./download_models.sh 8n640 10n320 11s480-opt
```
If you are interested in learning to compile the models on your own, refer to other MemryX Examples for instructions.

### Step 2: Run the Script/Program

To run this application, it is important to understand the command-line arguments and their significance.

* `-m`, `--model`: (Optional) Name of the model. Default: 8s640. 20 models are explicitly supported, and the following values are valid arguments.
  - `8n640` = YOLOv8 nano, 640 resolution
  - `8s640` = YOLOv8 small, 640 resolution
  - `8m640` = YOLOv8 medium, 640 resolution
  - `9t640` = YOLOv9 tiny, 640 resolution
  - `9s640` = YOLOv9 small, 640 resolution
  - `9m640` = YOLOv9 medium, 640 resolution
  - `10n320` = YOLOv10 nano, 320 resolution
  - `10n480` = YOLOv10 nano, 480 resolution
  - `10s320` = YOLOv10 small, 320 resolution
  - `10s480` = YOLOv10 small, 480 resolution
  - `10m320` = YOLOv10 medium, 320 resolution
  - `10m480` = YOLOv10 medium, 480 resolution
  - `11n320` = YOLOv11 nano, 320 resolution
  - `11n480-opt` = YOLOv11 nano, 480 resolution, MXA-optimized
  - `11n640-opt` = YOLOv11 nano, 640 resolution, MXA-optimized
  - `11n800-opt` = YOLOv11 nano, 800 resolution, MXA-optimized
  - `11s320` = YOLOv11 small, 320 resolution
  - `11s480-opt` = YOLOv11 small, 480 resolution, MXA-optimized
  - `11s640-opt` = YOLOv11 small, 640 resolution, MXA-optimized
  - `11m320` = YOLOv11 medium, 320 resolution
* `--cpu_decode`: (Optional) Use CPU for decoding instead of hardware-accelerated decoding. Applies to all input streams.
* `--video_paths`: (Optional) If using video files for input, specify as a space separated list formatted as path0,codec0 path1,codec1 ...
* `--usb_paths`: (Optional) If using USB cameras as input, specify as a space-separated list formatted as /dev/video0 /dev/video2 ...
* `--ip_paths`: (Optional) If using IP camera inputs, specify as space-separated list formatted as URL0,codec0 URL1,codec1 ...
* `--system`: Which type of host system is being used. Default: intel. Valid options: [`intel`, `amd`, `rockchip`, `broadcom`]

NOTE: For unsupported platform/codec configurations, the application will fall back to software/CPU decoding by default. Streams for which the software decoding fallback is triggered will be identifiable via application-generated warnings printed in the terminal. Refer to the following table for which platform/codec combinations support hardware-accelerated decoding:

|          | MJPEG | H.264 | H.265 |
|----------|-------|-------|-------|
| Intel    | Y     | Y     | Y     |
| AMD      | Y     | Y     | Y     |
| Rockchip | N     | Y     | Y     |
| Broadcom | N     | N     | Y     |

When using a combination of video files and camera inputs, the inference pipeline will be terminated once the shortest video file is parsed to completion. Allow the application about 10 seconds for proper shutdown.

Use the following commands to build the C++ executable:

```bash
cd src/
mkdir build && cd build
cmake ..
make -j
```

This will generate an executable called `main` in the build directory. Run the application as follows:

```bash
# Run with 3 USB camera streams as input on an Intel host:
./main -m 8s640 --usb_paths /dev/video0 /dev/video2 /dev/video4 --system intel

# Run with 2 IP camera streams as input, with one stream using H.264 encoding and the other using H.265/HEVC encoding on an AMD host:
./main -m 10n320 --ip_paths rtsp://user:psswd@xxx.xxx.x.xx:port,h264 rtsp://user:psswd@yyy.yyy.y.yy:port,h265 --system amd

# Run with a single H.264-encoded video file as input and use software/CPU decoding on a Rockchip host:
./main -m 11s480-opt --video_paths path/to/file.mp4,h264 --cpu_decode --system rockchip

# Run with a combination of video files, USB cams, and IP cams as input sources on a Broadcom host:
./main -m 9t640 --video_paths path/to/file.mp4,h264 --usb_paths /dev/video0 /dev/video2 --ip_paths rtsp://user:psswd@xxx.xxx.x.xx:port,h264 --system broadcom
```

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Models**: All YOLO model variants sourced from [Ultralytics Github](https://github.com/ultralytics/ultralytics)
  - License: [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗
