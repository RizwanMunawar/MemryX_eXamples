#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <csignal>
#include <queue>
#include <condition_variable>
#include <opencv2/opencv.hpp>

#include "memx/accl/MxAccl.h"
#include <memx/mxutils/gui_view.h>
#include "PoseApp.h"
#include "CartoonApp.h"

// Frame queues and sync
std::queue<cv::Mat> cartoon_queue;
std::queue<cv::Mat> pose_queue;
std::mutex cartoon_mutex, pose_mutex;
std::condition_variable cartoon_cv, pose_cv;

// Runtime state
std::atomic_bool runflag(true);
cv::VideoCapture vcap;

// Signal handler to safely stop
void signalHandler(int signum) {
    runflag.store(false);
}

const size_t MAX_QUEUE_SIZE = 20;  // You can adjust this as needed

void frameProducer(cv::VideoCapture& cap, std::atomic_bool& runflag, bool is_video) {
    cv::Mat frame;

    while (runflag.load()) {
        if (!cap.read(frame)) {
            std::cerr << "[Producer] End of video or failed to read frame.\n";
            break;
        }

        if (is_video) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30)); // Simulate ~30 FPS for video files
        }

        // --- Push to Cartoon Queue ---
        {
            std::lock_guard<std::mutex> lock(cartoon_mutex);
            cartoon_queue.push(frame.clone());
            cartoon_cv.notify_one();
        }

        // --- Push to Pose Queue ---
        {
            std::lock_guard<std::mutex> lock(pose_mutex);
            pose_queue.push(frame.clone());
            pose_cv.notify_one();
        }
    }

    cartoon_cv.notify_all();
    pose_cv.notify_all();
}


int main(int argc, char* argv[]) {
    bool use_cam = false;
    std::string video_path;
    std::string dfp_cartoon = "../../models/Facial_cartoonizer_512_512_3_onnx.dfp";
    std::string dfp_pose = "../../models/YOLO_v8_small_pose_640_640_3_onnx.dfp";
    std::string pose_post = "../../models/YOLO_v8_small_pose_640_640_3_onnx_post.onnx";

    // Init GUI
    MxQt gui(argc, argv);
    gui.screens[0]->SetSquareLayout(2);  // Two screens: Cartoon + Pose

    // Parse input arguments
    if (argc >= 2) {
        // Create a vector of strings for easier iteration
        std::vector<std::string> args(argv, argv + argc);
    
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--cam") {
                use_cam = true;
            } else if (args[i] == "--video" && i + 1 < args.size()) {
                video_path = args[++i];
            } else if (args[i] == "--dfp_cartoon" && i + 1 < args.size()) {
                dfp_cartoon = args[++i];
            } else if (args[i] == "--dfp_pose" && i + 1 < args.size()) {
                dfp_pose = args[++i];
            } else if ((args[i] == "--post" || args[i] == "-post") && i + 1 < args.size()) {
                pose_post = args[++i];
            } else {
                std::cout << "Unknown or incomplete argument: " << args[i] << "\n";
                std::cout << "Usage: ./cartoon_pose [--cam] [--video <path>] [--dfp_cartoon <path>] [--dfp_pose <path>] [--post <path>]\n";
                return -1;
            }
        }
    
        // Validation: Ensure we have an input source
        if (!use_cam && video_path.empty()) {
            std::cout << "Error: Must specify either --cam or --video <path>\n";
            return -1;
        }
    
    } else {
        std::cout << "Usage: ./cartoon_pose [--cam | --video <path>]\n";
        return -1;
    }

    signal(SIGINT, signalHandler);

    if (use_cam) {
        vcap.open(0, cv::CAP_V4L2);
    } else {
        vcap.open(video_path);
    }

    if (!vcap.isOpened()) {
        std::cerr << "[Main] Failed to open input source.\n";
        return -1;
    }

    // Accelerator options
    MX::RPC::SchedulerOptions sched_opts;
    sched_opts.frame_limit = 10;  // Max frames before DFP swap
    
    MX::RPC::ClientOptions client_opts{
        true,  // smoothing: Enable FPS smoothing
        30.0f  // fps_target: Limit input pacing to 30 FPS
    };

    // Cartoon pipeline
    std::vector<int> cartoon_device = {0};
    MX::Runtime::MxAccl accl_cartoon(dfp_cartoon, cartoon_device, {false, false}, false, sched_opts, client_opts);
    CartoonApp cartoonApp(&accl_cartoon, &cartoon_queue, &cartoon_mutex, &cartoon_cv, &runflag, &gui, 0);

    // Pose pipeline
    std::vector<int> pose_device = {0};
    MX::Runtime::MxAccl accl_pose(dfp_pose, pose_device, {true, true}, false, sched_opts, client_opts);
    PoseApp poseApp(&accl_pose, pose_post, &pose_queue, &pose_mutex, &pose_cv, &runflag, &gui, 1);

    // std::thread producer_thread(frameProducer, std::ref(vcap), std::ref(runflag));
    std::thread producer_thread(frameProducer, std::ref(vcap), std::ref(runflag), !use_cam);

    accl_pose.start();
    accl_cartoon.start();

    gui.Run();
    std::cout << "[Main] GUI exited. Shutting down..." << std::endl;

    runflag.store(false);

    accl_cartoon.wait();
    accl_pose.wait();

    accl_pose.stop();
    accl_cartoon.stop();

    producer_thread.join();

    return 0;
}
