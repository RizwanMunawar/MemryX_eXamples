#include <iostream>
#include <thread>
#include <signal.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "memx/accl/MxAccl.h"
#include <memx/accl/utils/blocky_queue.h>
#include <memx/mxutils/gui_view.h>
#include "MXVideoDecoder.h"
#include "yolov8.h"

extern "C" {
    #include <libavutil/time.h>
}

namespace fs = std::filesystem;

// Forward declaration
class YoloApp;

std::atomic_bool runflag;  // Atomic flag to control run state
bool is_running = true;

// YoloV8 application specific parameters
std::string model_name = "8s640";
fs::path model_path = "../../assets/models/8s640/model.dfp";  // Default model path
fs::path post_path = "../../assets/models/8s640post.onnx";  // Default post model path
std::string system_type = "amd"; // Default system type
int num_streams = 0;  // Number of parallel streams
std::string codec = "h264";  // Codec type for input video stream
std::vector<int> device_ids = {0};  // List of device IDs to use
std::vector<float> FPS_per_stream;
std::vector<double> CPU_load;
int str_start_idx = 0;
int str_end_idx = num_streams;

struct ChannelObject
{
    uint32_t disp_width;
    uint32_t disp_height;
    DisplayScreen *screen;
    std::vector<cv::Mat*> orig_frame_storage;
    MX::Utils::BlockyQueue<cv::Mat*> *orig_frame_freelist;      // Max 5 frames
    MX::Utils::BlockyQueue<cv::Mat*> raw_frame_queue{5};      // Max 5 frames
    MX::Utils::BlockyQueue<cv::Mat*> processed_frame_queue{5}; // Max 5 frames
    MXVideoDecoder* decoder_handle;
    YoloApp* yolo_handle;
};

struct InputSource
{
    std::string path;
    std::string codec; // "mjpeg", "h264", "h265"
    std::string type; // "USB", "IP", or "FILE"
};

// Global variables
InputSource input_sources[100];

ChannelObject g_chan_objs[100];

#define FPS_LOG_INTERVAL 120  // print out FPS every X frames
#define FRAME_QUEUE_MAX_LENGTH     5

// Signal handler to gracefully stop the program on SIGINT (Ctrl+C)
void signal_handler(int p_signal) {
    runflag.store(false);  // Stop the program
    is_running = false;
}

// Function to display usage information
void print_usage(const std::string& program_name) {
    std::cout << "Usage: " << program_name
              << " [-m <model_name>] [--cpu_decode] [--video_paths] [--usb_paths] [--ip_paths] \n"
              << "Options:\n"
              << "  -m, --model_name      (Optional) Name of the model. Default: 8s640. Valid options: [8s640, 8m640, 9s640, 9m640, 11s320, 11s480, 11m320]\n"
              << "  --cpu_decode          (Optional) Use CPU for decoding instead of hardware-accelerated decoding. Default: false\n"
              << "  --video_paths         If using video files for input, specify as a space-separated list formatted as path0,codec0 path1,codec1 ...\n"
              << "  --usb_paths           If using USB cameras as input, specify as a space-separated list formatted as /dev/video0 /dev/video2 ...\n"
              << "  --ip_paths            If using IP camera inputs, specify as space-separated list formatted as URL0,codec0 URL1,codec1 ...\n"
              << "  --system              Which type of host system is being used. Default: intel. Valid options: [intel, amd, rockchip, broadcom]\n";
}

pair<long, long> get_cpu_times() {
    ifstream stat_file("/proc/stat");
    string line;
    getline(stat_file, line);
    stat_file.close();

    vector<long> times;
    string cpu;
    long value;
    istringstream iss(line);
    iss >> cpu;
    while (iss >> value) {
        times.push_back(value);
    }

    long idle_time = times[3]; //idle time is 4th field
    long total_time = accumulate(times.begin(), times.end(), 0L);

    return {idle_time, total_time};
}

double calculate_cpu_load(pair<long, long> prev, pair<long, long> curr) {
    long idle_diff = curr.first - prev.first;
    long total_diff = curr.second - prev.second;

    return 100 * (1.0 - static_cast<double>(idle_diff) / total_diff);
}

void InfoWatcher(int monitoring_duration_seconds, MX::Runtime::MxAccl* accl) {
    // Use this function to measure CPU load
    pair<long, long> prev_times = get_cpu_times();
    int run_count = 0;
    unsigned int sleep_duration_ms = 1000;    
    int target_count = monitoring_duration_seconds * 1000 / sleep_duration_ms;
    while (is_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration_ms));
        run_count++;
        if (run_count == target_count) {
            {
                pair<long, long> curr_times = get_cpu_times();
                double cpu_load = calculate_cpu_load(prev_times, curr_times);
                CPU_load.push_back(cpu_load);
                prev_times = curr_times;
            }
            run_count = 0;
        }
    }
}

void RunDecoding(int idx) {
    auto &chan_obj = g_chan_objs[idx];
    cv::Mat *disp_frame;
    AVFrame* rgb_frame = av_frame_alloc();
    rgb_frame->format = AV_PIX_FMT_RGB24;
    rgb_frame->width = chan_obj.disp_width;
    rgb_frame->height = chan_obj.disp_height;
    uint32_t disp_frame_size = chan_obj.disp_width * chan_obj.disp_height * 3;
    const double target_fps = 30.0;
    const int64_t frame_interval = 1000000 / target_fps;
    int64_t next_pts = av_gettime_relative() + frame_interval;

    while (is_running) {
        if (!chan_obj.decoder_handle->is_running) {
            is_running = false;
            break;
        }
        
        rgb_frame = chan_obj.decoder_handle->get_frame();
        
        // Check if frame is valid
        if (!rgb_frame || !rgb_frame->data[0]) {
            std::cout << "EOF: Invalid frame received from decoder" << std::endl;
            is_running = false;
            break;
        }

        disp_frame = new cv::Mat(chan_obj.disp_height, chan_obj.disp_width, CV_8UC3, rgb_frame->data[0], rgb_frame->linesize[0]);

        if (!chan_obj.raw_frame_queue.push_timeout(disp_frame, 10000)) {
            delete disp_frame; // Only delete if push failed
            is_running = false;
            break;
        }
        int64_t now = av_gettime_relative();
        if (next_pts > now) {
            av_usleep(next_pts - now);
        }
        next_pts += frame_interval;
    }
    runflag.store(false);
    return;
}

class YoloApp {
    private:
        // Application Variables
        std::vector<std::unique_ptr<MX::Utils::BlockyQueue<cv::Mat*>>> frames_queues;

        std::vector<cv::Mat> pre_frame_storage;
        MX::Utils::BlockyQueue<cv::Mat*> *pre_frame_freelist;  // Freelist for pre-allocated frames
        
        // FPS related
        std::vector<int> frame_count;  // Per-stream frame count
        int total_frame_count = 0;
        std::vector<float> stream_fps;  // Per-stream FPS
        float total_fps_number = .0;  // Total FPS across all streams
        std::vector<float> history_fps;
        std::vector<std::chrono::milliseconds> stream_start_ms;  // Per-stream start time
        
        MX::Types::MxModelInfo model_info;  // Model info structure
        std::vector<float*> mxa_output;  // Buffer for the output of the accelerator

        YOLOv8* yolov8;  // YOLOv8 model

        // Input callback function to fetch frames and preprocess them
        bool incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int stream_idx) {
            // std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
            auto &chan_obj = g_chan_objs[stream_idx];
            if (runflag.load()) {
                cv::Mat* rgbImage = chan_obj.orig_frame_freelist->pop(); // removed alloc
                cv::Mat* preProcFrame = pre_frame_freelist->pop(); // removed alloc

                while(true){
                    if (chan_obj.raw_frame_queue.pop_timeout(rgbImage, 2000)) {
                        frames_queues[stream_idx]->push(rgbImage);
                    } else {
                        if(!chan_obj.raw_frame_queue.pop_timeout(rgbImage, 10000)) {
                            break;
                        }
                        runflag.store(false);
                        return false;
                    }
                    *preProcFrame = yolov8->preprocess(*rgbImage);
                    dst[0]->set_data((float*)preProcFrame->data);
                    pre_frame_freelist->push(preProcFrame); // return to freelist
                    return true;
                }
                runflag.store(false);
                return false;
            }
            else {
                return false;
            }
        }

        // Output callback function to process MXA output and display results
        bool outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int stream_idx) {
            auto &chan_obj = g_chan_objs[stream_idx];
            for (int i = 0; i < model_info.num_out_featuremaps; ++i) {
                src[i]->get_data(mxa_output[i]);
            }

            cv::Mat* original_frame_ptr = nullptr;
            
            if (!frames_queues[stream_idx]->pop_timeout(original_frame_ptr, 1000)) {
                std::cout << "ERROR: frames_queue timeout in outcall!" << std::endl;
                return false;
            }
            
            // Set confidence threshold and process detection results
            YOLOv8Result result;
            yolov8->postprocess(mxa_output, result);
            yolov8->draw_result(result, *original_frame_ptr);

            // Calculate per-stream FPS and total FPS
            frame_count[stream_idx]++;
            total_frame_count++;
            if (frame_count[stream_idx] == 1) {
                stream_start_ms[stream_idx] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            }
            else if (frame_count[stream_idx] % FPS_LOG_INTERVAL == 0) {
                std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - stream_start_ms[stream_idx];
                stream_fps[stream_idx] = (float)FPS_LOG_INTERVAL * 1000 / (float)(duration.count());
                frame_count[stream_idx] = 0;
                
                // Calculate total FPS across all streams
                total_fps_number = 0.0f;
                for (int s = 0; s < stream_fps.size(); ++s) {
                    total_fps_number += stream_fps[s];
                }
                
                std::cout << std::fixed << std::setprecision(1);
                std::cout << "Stream " << stream_idx << " - FPS: " << stream_fps[stream_idx] 
                          << " | Total_frame_count: " << total_frame_count 
                          << " | Total_FPS: " << total_fps_number << '\n';
                history_fps.push_back(total_fps_number);
            }

            chan_obj.screen->SetDisplayFrame(stream_idx, original_frame_ptr, stream_fps[stream_idx]);
            // Return the original frame pointer to freelist instead of deleting
            chan_obj.orig_frame_freelist->push(original_frame_ptr);

            return true;
        }

    public:
        float get_avg_fps() const
        {
            float sum = 0;
            for (const auto& fps : history_fps)
            {
                sum += fps;
            }
            return sum / history_fps.size();
        }

        // Constructor to initialize YOLOv8 object
        YoloApp(MX::Runtime::MxAccl* accl_, int num_streams, int disp_width, int disp_height) {
            frames_queues.resize(num_streams);
            for (int i = 0; i < num_streams; i++) {
                frames_queues[i] = std::make_unique<MX::Utils::BlockyQueue<cv::Mat*>>(FRAME_QUEUE_MAX_LENGTH);
            }
            
            // Initialize per-stream FPS tracking
            frame_count.resize(num_streams, 0);
            stream_fps.resize(num_streams, 0.0f);
            stream_start_ms.resize(num_streams);
            int input_image_width = disp_width;
            int input_image_height = disp_height;

            model_info = accl_->get_model_info(0 /* model_id */);

            // compute padding
            yolov8 = new YOLOv8(model_info.in_featuremap_shapes);
            yolov8->compute_padding(input_image_width, input_image_height);

            pre_frame_freelist = new MX::Utils::BlockyQueue<cv::Mat*>(num_streams);
            pre_frame_storage.clear();
            for(int i=0; i < num_streams; i++){
                cv::Mat *newmat = new cv::Mat(cv::Size(model_info.in_featuremap_shapes[0][0],
                                                       model_info.in_featuremap_shapes[0][1]), CV_8UC3);
                pre_frame_storage.push_back(*newmat);
                pre_frame_freelist->push(newmat);
            }

            // Get model info and allocate output buffer
            mxa_output.resize(model_info.num_out_featuremaps);
            for (int i = 0; i < model_info.num_out_featuremaps; ++i) {
                mxa_output[i] = new float[model_info.out_featuremap_sizes[i]];
            }

            // Bind input/output callback functions
            auto in_cb = std::bind(&YoloApp::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
            auto out_cb = std::bind(&YoloApp::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);

            // Connect streams to the accelerator
            for (int i = 0; i < num_streams; i++) {
                accl_->connect_stream(in_cb, out_cb, i /**Unique Stream Idx */, 0 /**Model Idx */);
            }

            // Start the input/output streams
            runflag.store(true);
        }

        ~YoloApp() {
            delete yolov8;
            for (int i = 0; i < static_cast<int>(mxa_output.size()); ++i) {
                delete[] mxa_output[i];
            }
            mxa_output.clear();  // Clean up memory
        }
};


int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);  // Set up signal handler
    std::string video_str = "/dev/video0";
    std::string codec = "h264";
    bool sw_decode = false;
    cv::setNumThreads(2);
    MxQt gui(argc, argv);
    int screen_idx = 0;
    DisplayScreen *screen = gui.screens.at(screen_idx);



    // Iterate through the arguments
    for (int i = 1; i < argc; i++) {

        std::string arg = argv[i];

        // Handle -d or --dfp_path
        if (arg == "-m" || arg == "--model_name") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                model_name = argv[++i];
                if (model_name == "8s640") {
                    model_path = "../../assets/models/8s640/model.dfp";
                    post_path = "../../assets/models/8s640/post.onnx";
                } else if (model_name == "8n640") {
                    model_path = "../../assets/models/8n640/model.dfp";
                    post_path = "../../assets/models/8n640/post.onnx";
                } else if (model_name == "8m640") {
                    model_path = "../../assets/models/8m640/model.dfp";
                    post_path = "../../assets/models/8m640/post.onnx";
                } else if (model_name == "9t640") {
                    model_path = "../../assets/models/9t640/model.dfp";
                    post_path = "../../assets/models/9t640/post.onnx";
                } else if (model_name == "9s640") {
                    model_path = "../../assets/models/9s640/model.dfp";
                    post_path = "../../assets/models/9s640/post.onnx";
                } else if (model_name == "9m640") {
                    model_path = "../../assets/models/9m640/model.dfp";
                    post_path = "../../assets/models/9m640/post.onnx";
                } else if (model_name == "10n320") {
                    model_path = "../../assets/models/10n320/model.dfp";
                    post_path = "../../assets/models/10n320/post.onnx";
                } else if (model_name == "10n480") {
                    model_path = "../../assets/models/10n480/model.dfp";
                    post_path = "../../assets/models/10n480/post.onnx";
                } else if (model_name == "10s320") {
                    model_path = "../../assets/models/10s320/model.dfp";
                    post_path = "../../assets/models/10s320/post.onnx";
                } else if (model_name == "10s480") {
                    model_path = "../../assets/models/10s480/model.dfp";
                    post_path = "../../assets/models/10s480/post.onnx";
                } else if (model_name == "10m320") {
                    model_path = "../../assets/models/10m320/model.dfp";
                    post_path = "../../assets/models/10m320/post.onnx";
                } else if (model_name == "10m480") {
                    model_path = "../../assets/models/10m480/model.dfp";
                    post_path = "../../assets/models/10m480/post.onnx";
                } else if (model_name == "11n320") {
                    model_path = "../../assets/models/11n320/model.dfp";
                    post_path = "../../assets/models/11n320/post.onnx";
                } else if (model_name == "11n640-opt") {
                    model_path = "../../assets/models/11n640-opt/model.dfp";
                    post_path = "../../assets/models/11n640-opt/post.onnx";
                } else if (model_name == "11n800-opt") {
                    model_path = "../../assets/models/11n800-opt/model.dfp";
                    post_path = "../../assets/models/11n800-opt/post.onnx";
                } else if (model_name == "11s320") {
                    model_path = "../../assets/models/11s320/model.dfp";
                    post_path = "../../assets/models/11s320/post.onnx";
                } else if (model_name == "11m320") {
                    model_path = "../../assets/models/11m320/model.dfp";
                    post_path = "../../assets/models/11m320/post.onnx";
                } else {
                    std::cerr << "Invalid model name specified\n";
                    print_usage(argv[0]);
                    return 1;
                }
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                print_usage(argv[0]);
                return 1;
            }
        }
        else if(arg == "-c" || arg == "--codec") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                codec = argv[++i];
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                print_usage(argv[0]);
                return 1;
            }
        }
        else if (arg == "--cpu_decode") {
            sw_decode = true;
        }
        else if (arg == "--video_paths") {
            // Collect all path-codec pairs until next option or end
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string entry = argv[++i];
                auto comma_pos = entry.find(',');
                if (comma_pos == std::string::npos) {
                    std::cerr << "Invalid video_paths entry: " << entry << ". Expected format: path0,codec0 path1,codec1 ...\n";
                    print_usage(argv[0]);
                    return 1;
                }
                std::string path = entry.substr(0, comma_pos);
                std::string entry_codec = entry.substr(comma_pos + 1);
                input_sources[num_streams].path = path;
                input_sources[num_streams].codec = entry_codec;
                input_sources[num_streams].type = "FILE";
                num_streams++;
            }
        }
        else if (arg == "--usb_paths") {
            // Collect all USB device paths until next option or end
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                input_sources[num_streams].path = argv[++i];
                input_sources[num_streams].codec = "mjpeg";
                input_sources[num_streams].type = "USB";
                num_streams++;
            }
        }
        else if (arg == "--ip_paths") {
            // Collect all IP path-codec pairs until next option or end
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string entry = argv[++i];
                auto comma_pos = entry.find(',');
                if (comma_pos == std::string::npos) {
                    std::cerr << "Invalid ip_paths entry: " << entry << ". Expected format: url0,codec0 url1,codec1 ...\n";
                    print_usage(argv[0]);
                    return 1;
                }
                std::string url = entry.substr(0, comma_pos);
                std::string entry_codec = entry.substr(comma_pos + 1);
                input_sources[num_streams].path = url;
                input_sources[num_streams].codec = entry_codec;
                input_sources[num_streams].type = "IP";
                num_streams++;
            }
        }
        else if (arg == "--system") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                system_type = argv[++i];
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                print_usage(argv[0]);
                return 1;
            }
        }
        // Handle unknown options
        else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }


    screen->SetSquareLayout(num_streams);

    // Init Accl object
    std::array<bool, 2> use_model_shape = {false, false};
    bool local_mode = true; // Running in local mode as we are not using MXA manager for this sample app
    MX::Runtime::MxAccl accl{fs::path(model_path), device_ids, use_model_shape, local_mode};

    int disp_width = screen->GetViewerWidth(screen_idx);
    int disp_height = screen->GetViewerHeight(screen_idx);

    // Create the one and only YoloApp Object
    YoloApp* yolo_handle = new YoloApp(&accl, num_streams, disp_width, disp_height);

    // Create buffers for each stream and initialize decoder objects
    std::vector<std::thread> threads;
    for (int i = 0; i < num_streams; ++i) {

        DecoderConfig config;
        std::string this_video_str = video_str;
        std::string this_codec = codec;

        // Use input_sources array if populated
        if (!input_sources[i].path.empty()) {
            this_video_str = input_sources[i].path;
            this_codec = input_sources[i].codec;
            
            if (input_sources[i].type == "FILE") {
                config.source_type = DecoderConfig::FILE;
            } else if (input_sources[i].type == "USB") {
                config.source_type = DecoderConfig::USB;
            } else if (input_sources[i].type == "IP") {
                config.source_type = DecoderConfig::IP;
            }
        }

        if(this_codec == "h264") {
            if (system_type == "intel" || system_type == "amd") {
                config.device_type = "vaapi";
                config.decoder_name = "h264";
                config.force_sw = sw_decode;
            }
            else if (system_type == "rockchip") {
                config.device_type = "rkmpp";
                config.decoder_name = "h264_rkmpp";
                config.force_sw = sw_decode;
            }
            else {
                config.device_type = "drm";
                config.decoder_name = "h264";
                config.force_sw = true; // Hardware h264 decoding not supported on Broadcom
                std::cout << "Warning: Forcing software decoding for h264 on Broadcom platforms\n";
            }
        }
        else if(this_codec == "h265") {
            if (system_type == "intel" || system_type == "amd") {
                config.device_type = "vaapi";
                config.decoder_name = "hevc";
                config.force_sw = sw_decode;
            }
            else if (system_type == "rockchip") {
                config.device_type = "rkmpp";
                config.decoder_name = "hevc_rkmpp";
                config.force_sw = sw_decode;
            }
            else {
                config.device_type = "drm";
                config.decoder_name = "hevc";
                config.force_sw = sw_decode;
            }
        }
        else if (this_codec ==  "mjpeg") {
            if (system_type == "intel" || system_type == "amd") {
                config.device_type = "vaapi";
                config.decoder_name = "mjpeg";
                config.force_sw = sw_decode;
            }
            else if (system_type == "rockchip") {
                config.device_type = "rkmpp";
                config.decoder_name = "mjpeg";
                config.force_sw = true; // Hardware mjpeg decoding not supported on Rockchip/Broadcom
                std::cout << "Warning: Forcing software decoding for mjpeg on Rockchip platforms\n";
            }
            else {
                config.device_type = "drm";
                config.decoder_name = "mjpeg";
                config.force_sw = true; // Hardware mjpeg decoding not supported on Broadcom
                std::cout << "Warning: Forcing software decoding for mjpeg on Broadcom platforms\n";
            }
        }
        else {
            std::cerr << "Invalid codec specified for stream " << i << ": " << this_codec << "\n";
            print_usage(argv[0]);
            return 1;
        }

        config.disp_width = disp_width;                       // Display width
        config.disp_height = disp_height;                     // Display height
        config.access_str = this_video_str;

        g_chan_objs[i].screen = screen;
        g_chan_objs[i].disp_width = disp_width;   // Use the same as config.disp_width
        g_chan_objs[i].disp_height = disp_height; // Use the same as config.disp_height

        g_chan_objs[i].orig_frame_freelist = new MX::Utils::BlockyQueue<cv::Mat*>(5);
        g_chan_objs[i].orig_frame_storage.clear();
        for (int j = 0; j < 5; j++) {
            cv::Mat *newmat = new cv::Mat(disp_height, disp_width, CV_8UC3);
            g_chan_objs[i].orig_frame_storage.push_back(newmat);
            g_chan_objs[i].orig_frame_freelist->push(newmat);
        }

        g_chan_objs[i].decoder_handle = new MXVideoDecoder(config); // Create decoder handle for the stream
        g_chan_objs[i].decoder_handle->initiate_decoder();

        threads.emplace_back(RunDecoding, i); // Run decoding for the stream in its own thread

        g_chan_objs[i].yolo_handle = yolo_handle; // All streams share the same YoloApp instance
    }

    runflag.store(true);

    std::thread info_watcher = std::thread(InfoWatcher, 1 /*monitoring duration in seconds*/, &accl);

    // Run the accelerator and wait
    accl.start();

    screen->show();

    gui.Run();

    is_running = false;
    runflag.store(false);  // Stop the application

    info_watcher.join();

    accl.stop();

    for (auto &thread : threads) {
        if (thread.joinable()) thread.join();
    }

    // clean up all the resources
    for (int i = 0; i < num_streams; i++) {
        auto &chan_obj = g_chan_objs[i];
        chan_obj.decoder_handle->clean_up_decoder();

        // Clean up original frame storage
        for (int j = 0; j < 5; j++) {
            delete chan_obj.orig_frame_storage[j];
            chan_obj.orig_frame_storage[j] = nullptr;
        }
        
        delete chan_obj.orig_frame_freelist;
        chan_obj.orig_frame_freelist = nullptr;
    }

    // Calculate final average FPS
    float avg_fps = 0.0f;
    float final_fps = 0.f;
    final_fps = g_chan_objs[0].yolo_handle->get_avg_fps();
    avg_fps = final_fps / num_streams;

    // Calculate average CPU load
    double total_cpu = 0.0;
    for (int i = 0; i < CPU_load.size(); ++i) {
        total_cpu += CPU_load[i];
    }
    double avg_cpu = total_cpu / CPU_load.size();

    // Print final statistics
    std::cout << "\n\n================ Final Statistics ================\n";
    std::cout << "Average FPS across all streams: " << avg_fps << "\n";
    std::cout << "Average CPU Load: " << avg_cpu << "%\n";
    std::cout << "===================================================\n\n";

    // Cleanup
    delete yolo_handle;
}
