# Object Detection Using Yolo11 with MxPrepost Library

The **Object Detection** example demonstrates real-time object detection using the pre-trained YOLO11s model with [MxPrepost](https://developer.memryx.com/api/mxprepost.html) for faster inference with MemryX MX3.

<p align="center">
  <img src="assets/objectDetection_yolo11s.png" alt="Object Detection Example" width="45%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Yolo11s](https://docs.ultralytics.com/models/yolo11/)                                            |
| **Model Type**       | Object Detection                                                      |
| **Framework**        | [onnx](https://onnx.ai/)                                                   |
| **Model Source**     | [Download from Ultralytics GitHub or docs](https://docs.ultralytics.com/models/yolo11/) and export to onnx |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p2/YOLO11_small_640_640_3_onnx.zip)   |
| **Model Resolution** | 640x640                                                 |
| **Output**           | Bounding box coordinates with objectness score, and class probabilities |
| **OS**               | Linux |
| **License**          | [AGPL](LICENSE.md)                                       |

## Requirements

Before running the application, ensure that OpenCV and mxprepost pip packages are installed. You can install do so with the following command:

```bash
# For application
pip install --extra-index-url https://developer.memryx.com/pip opencv-python==4.11.0.86 "mxprepost~=2.2.0"
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/2p2/YOLO11_small_640_640_3_onnx.zip
mkdir -p models
unzip YOLO11_small_640_640_3_onnx.zip -d models
```

<details>
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained YOLO11s model and export it to ONNX:

You can use the following code to download the pre-trained yolo11s.pt model and export it to ONNX format:

```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")  # load an official model

# Export the model

# ONNX format
model.export(format="onnx")
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -m yolo11s.onnx -v --autocrop -c 4
```
The compiler will generate the DFP file and a post-processing file. However, we will not be using the cropped onnx layers in this tutorial, and will be using MxPrepost instead. So you only need the `.dfp`.

</details>

### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference using Python and mxprepost:

```bash
# ensure a camera device is connected as default video input is a cam
cd src/python/
python run_objectiondetection.py
```
You can specify the DFP (Compiled Model) path using the following options.

* `-d` or `--dfp`: Path to the compiled DFP file (default is models/YOLO11_small_640_640_3_onnx.dfp)

You can specify the input video path with the following option:

* `--video_paths` : Paths to video files as inputs (default is /dev/video0, which is typically the first USB cam connected)

For example, to run with a specific video and DFP file, use:

```bash
python run_objectiondetection.py -d <dfp_path> --video_paths /dev/video0
```

If no arguments are provided, the script will use the default post-processing model and DFP paths.

## Tutorial

A more detailed tutorial with complete code explanations is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/tutorials/realtime_inf/mxprepost.html)


## Third-Party License

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Yolo11 from Ultralytics GitHub](https://docs.ultralytics.com/models/yolo11/) 🔗
  - License: [AGPL](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)  🔗

- **Preview Image**: ["Little Boy Lying in Bed with a Dog" on Pexels](https://www.pexels.com/photo/little-boy-lying-in-bed-with-a-corgi-dog-5264054/)
  - License: [Pexels License](https://www.pexels.com/license/)

