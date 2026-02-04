# Aimbot

This application uses YOLOv7 running on the MX3 to automatically point and click the mouse at "people" classes in Windows games. This guide provides setup instructions, model details, and code snippets to run this application.

**Warning**: this app is for hobbyist and personal entertainment purposes only, and should never be used to cheat in online games.

<p align="center">
  <img src="assets/aimbot_demo.gif" alt="Aimbot Example" width="55%" />
</p>


## Overview

| **Property**         | **Details**
|----------------------|------------------------------------------
| **Model**            | YOLOv7-tiny
| **Model Type**       | Object Detection
| **Framework**        | ONNX
| **Model Source**     | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
| **Pre-compiled DFP** | [aimbot.zip](https://developer.memryx.com/example_files/aimbot.zip)
| **Input Resolution** | 320x800
| **Output**           | Bounding Boxes & Class IDs
| **OS**               | Windows
| **License**          | [GPL](LICENSE.md)




## Requirements

Before continuing, please install `python3.11` for Windows, most easily done via the [Microsoft Store](https://apps.microsoft.com/detail/9nrwmjp3717k)🔗


## Running the Application

### Preparation

Next we need to create a Python virtual env and install dependencies.

Conveniently, you can just run the `setup_env.bat` script by opening the `src/` folder and double-clicking on it.


### Download/Compile the Model

Download the [aimbot.zip](https://developer.memryx.com/example_files/aimbot.zip) file and extract it into the `models/` folder.

Your folder structure should now be:

```
|- README.md
|- LICENSE.md
|- models/
|  |- yolov7-tiny_320_800.dfp
|  |- yolov7-tiny_320_800.onnx
|  |- yolov7-tiny_320_800.post.onnx
|
|- src/
|  |- 311env/
|     |- (venv data)
|  |- aimbot_mxa.py
|  |- aimbot_onnx.py
|  |- (etc.)

```


<details>
<summary>(Optional) Alternatively, Compile It Yourself</summary>
<br>

Instead of using the pre-made DFP, you can export the model from the [original source](https://github.com/WongKinYiu/yolov7) (on Linux/WSL) with the following usage of their `export.py` script:


```bash
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
  --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 320 800
```

Then you can compile with the MemryX NeuralCompiler with the following command:

```bash
mx_nc -v -m yolov7-tiny.onnx --autocrop
```

Then just move the files into the `models/` directory on your Windows machine and make sure to rename them accordingly. 

</details>


### Run


#### Start

To run, first start your target game/application. Please follow these guidelines for best results:

* Use a monitor with 1440p (2K) or 2160p (4K) resolution
* Set the game to 'Windowed' mode with 1920x1080 resolution
* Choose graphical settings that will leave a tiny bit of room for the GPU to do screen capture -- don't push to 100% GPU usage. `dxcam` in this app's code defaults to just 30 FPS capture rate, so you don't need to go much higher than 60 FPS in your game. A maxed-out GPU will greatly hurt capture rate.

Once you're ready, double-click the `run_mxa.bat` and select the window corresponding to your game.

A small window showing the detection results will open in the background, so that you can monitor the neural network's output. Boxes will not be drawn over the game window itself.

#### Auto Aim/Click

Once running, press the `CAPS_LOCK` key to toggle the aim-and-click functionality on and off.


### Settings

There are multiple configuration options that you may need to tweak for your system and/or game.

Before starting the aimbot application, you may edit the `src/aimbot_mxa.py` script to edit the config. Options include:

* `aaRightShift`: right-shift the center of the screen by the give number of pixels. Useful for some games.
* `aaMovementAmp`: control how rapidly the mouse accelerates when aiming. Note that the default value (0.80) might have issues depending on your Windows mouse acceleration settings.
* `aaQuitKey`: which keyboard key to bind the aimbot quit action to.
* `auto_click`: if False, when `CAPS_LOCK` is enabled the aimbot will just aim and not click.
* `headshot_mode`: adds a small vertical offset to try to aim for the head.
* `centerOfScreen`: when there's multiple targets, always choose the one closest to the center of the screen, instead of closest to the current pointer position.
* `visuals`: enable/disable the tiny window that shows neural network output


### Advanced Options

If you're looking to play around with the aimbot's internals, there are a few places that may be of interest:

* `dxcam.create` and `camera.start` in the `init_capture()` funtion
   - Here you can tweak [dxcam's settings](https://github.com/ra1nty/DXcam?tab=readme-ov-file#advanced-usage-and-remarks), such as capture FPS.
* `headshot_offset` in the `display_and_shoot()` function
   - These lines control the box offset amounts to move in order to target heads.
* `abs(mouseMove[0]) <= 2.5` in `display_and_shoot()`
   - This section of the code controls the click cooldown and the movement thresholds before a click event is sent.


### Discussion

#### Recommanded / Not Recommended Games

This app was tested with [Aimbeast](https://store.steampowered.com/app/1100990/Aimbeast/) in a simple "point and click" map.

Our aiming heuristics aren't advanced enough to handle games where you have to move your character around, i.e. the majority of First Person Shooter games 😅. Third-person perspective games also have issues, because the playable character is recognized as a target and causes the mouse to freak out and get stuck.


#### Extending / Future Work

The issues with player movement and third-person perspectives could likely be improved by an interested developer with the time to create more advanced control systems.

As a more advanced exercise, one might consider creating game-specific neural networks by training yolov7 on a custom dataset of images captured from the game. This could cover non-human enemies or even a "friendly" vs "foe" classification instead of just "person".


## Third-Party Licenses

* Model: Copyright (c) 2023 Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao, [GPL license](https://github.com/WongKinYiu/yolov7) 🔗
