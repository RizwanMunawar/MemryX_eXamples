#include <iostream>
#include <thread>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <chrono>
#include "memx/accl/MxAccl.h"

namespace fs = std::filesystem;

std::atomic_bool runflag;

//YoloV26 application specific parameters
fs::path model_path = "YOLO26_nano_640_640_3_onnx.dfp";
fs::path postprocessing_model_path = "YOLO26_nano_640_640_3_onnx_post.onnx";

#define FRAME_QUEUE_MAX_LENGTH     20
//signal handler
void signal_handler(int p_signal) {
    runflag.store(false);
}

// Function to display usage information
void printUsage(const std::string& programName) {
    std::cout << "Usage: " << programName
        << " [-d <dfp_path>] [-m <post_model>] [--video_paths \"cam:0,vid:video_path\"]\n"
        << "Options:\n"
        << "  -d, --dfp_path        (Optional) Path to the DFP. Default: " << model_path << "\n"
        << "  -m, --post_model      (Optional) Path to the post-model. Default: " << postprocessing_model_path << "\n"
        << "  --video_paths         (Optional) Video paths in the format \"cam:0,vid:video_path,vid:video2_path\". Default: cam:0\n";
}


//Struct to hold detection outputs
struct detectedObj {
    int x1;
    int x2;
    int y1;
    int y2;
    int obj_id;
    float accuracy;

    detectedObj(int x1_, int x2_, int y1_, int y2_, int obj_id_, float accuracy_) {
        x1 = x1_;
        x2 = x2_;
        y1 = y1_;
        y2 = y2_;
        obj_id = obj_id_;
        accuracy = accuracy_;
    }
};

// In case of cameras try to use best possible input configurations which are setting the
// resolution to 640x480 and try to set the input FPS to 30
bool configureCamera(cv::VideoCapture& vcap) {
    bool settings_success = true;

    try {
        if (!vcap.set(cv::CAP_PROP_FRAME_HEIGHT, 480) ||
            !vcap.set(cv::CAP_PROP_FRAME_WIDTH, 640) ||
            !vcap.set(cv::CAP_PROP_FPS, 30)) {
            std::cout << "Setting vcap Failed\n";
            cv::Mat simpleframe;
            if (!vcap.read(simpleframe)) {
                settings_success = false;
            }
        }
    }
    catch (...) {
        std::cout << "Exception occurred while setting properties\n";
        settings_success = false;
    }

    return settings_success;
}

// Tries to open the camera with custom settings set in configureCamera
// If not possible, open it with default settings
bool openCamera(cv::VideoCapture& vcap, int device, int api) {
    vcap.open(device, api);
    if (!vcap.isOpened()) {
        std::cerr << "Failed to open vcap\n";
        return false;
    }

    if (!configureCamera(vcap)) {
        vcap.release();
        vcap.open(device, api);
        if (vcap.isOpened()) {
            std::cout << "Reopened vcap with original resolution\n";
        }
        else {
            std::cerr << "Failed to reopen vcap\n";
            return false;
        }
    }

    return true;
}

class YoloV26 {
private:
    // Model Params
    int model_input_width;//width of model input image
    int model_input_height;//height of model input image
    int input_image_width;//width of input image
    int input_image_height;//height of input image
    int num_boxes = 300;//Maximum number of boxes that can be output by the YOLOv26 model
    float conf_thresh = 0.4;//Confidence threshold of the boxes
    std::vector<std::string> class_names = { //Class names list of COCO dataset
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    //Application Variables
    std::deque<cv::Mat> frames_queue;
    std::mutex frame_queue_mutex;
    int num_frames = 0;
    int frame_count = 0;
    std::chrono::milliseconds start_ms;
    cv::VideoCapture vcap;
    bool src_is_cam = false;
    std::vector<size_t> in_tensor_sizes;
    std::vector<size_t> out_tensor_sizes;
    MX::Types::MxModelInfo model_info;
    float* mxa_output;
    cv::Mat displayImage;
    std::string video_src;
    std::string window_name;

    cv::Mat preprocess(cv::Mat& image) {

        cv::Mat resizedImage;
        cv::dnn::blobFromImage(image, resizedImage, 1.0, cv::Size(model_input_width, model_input_height), cv::Scalar(0, 0, 0), true, false);

        // Convert image to float32 and normalize
        cv::Mat floatImage;
        resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

        return floatImage;
    }

    void draw_bounding_box(cv::Mat& image, std::vector<detectedObj>& detections_vector) {
        for (int i = 0; i < detections_vector.size(); ++i) {
            detectedObj detected_object = detections_vector[i];
            cv::rectangle(image, cv::Point(detected_object.x1, detected_object.y1), cv::Point(detected_object.x2, detected_object.y2), cv::Scalar(0, 255, 0), 2);

            cv::putText(image, class_names.at(detected_object.obj_id),
                cv::Point(detected_object.x1, detected_object.y1 - 3), cv::FONT_ITALIC,
                0.8, cv::Scalar(255, 255, 255), 2);

            cv::putText(image, std::to_string(detected_object.accuracy),
                cv::Point(detected_object.x1, detected_object.y1 + 30), cv::FONT_ITALIC,
                0.8, cv::Scalar(255, 255, 0), 2);
        }
    }

    std::vector<detectedObj> get_detections(float* output, int num_boxes) {
        std::vector<detectedObj> detections;

        // YOLOv26 output format: (1, 300, 6)
        // [x1, y1, x2, y2, confidence, class_id]
        for (int i = 0; i < num_boxes; i++) {
            float x1 = output[i * 6 + 0];
            float y1 = output[i * 6 + 1];
            float x2 = output[i * 6 + 2];
            float y2 = output[i * 6 + 3];
            float accuracy = output[i * 6 + 4];
            int classPrediction = static_cast<int>(output[i * 6 + 5]);

            if (accuracy < conf_thresh) {
                continue;
            }

            // Skip clearly invalid boxes
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            // Scale from model input space to original image space
            x1 = (x1 / model_input_width) * input_image_width;
            x2 = (x2 / model_input_width) * input_image_width;
            y1 = (y1 / model_input_height) * input_image_height;
            y2 = (y2 / model_input_height) * input_image_height;

            // Reject boxes completely outside the image
            if (x2 <= 0 || y2 <= 0 || x1 >= input_image_width || y1 >= input_image_height) {
                continue;
            }

            // Clip boxes to image boundaries
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(input_image_width - 1)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(input_image_height - 1)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(input_image_width - 1)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(input_image_height - 1)));

            // Reject boxes that became invalid after clipping
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            detectedObj obj(x1, x2, y1, y2, classPrediction, accuracy);
            detections.push_back(obj);
        }

        return detections;
    }

    bool incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel) {

        if (runflag.load()) {
            cv::Mat inframe;
            cv::Mat rgbImage;

            while (true) {
                bool got_frame = vcap.read(inframe);

                if (!got_frame) {
                    std::cout << "No frame \n\n\n";
                    vcap.release();
                    return false;  // return false if frame retrieval fails/stream is done sending input
                }
                if (src_is_cam && (frames_queue.size() >= FRAME_QUEUE_MAX_LENGTH)) {
                    // drop the frame and try again if we've hit the limit
                    continue;
                }
                else {
                    cv::cvtColor(inframe, rgbImage, cv::COLOR_BGR2RGB);
                    {
                        std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                        frames_queue.push_back(rgbImage);
                    }
                }

                // Preprocess frame
                cv::Mat preProcframe = preprocess(rgbImage);
                // Set preprocessed input data to be sent to accelarator
                dst[0]->set_data((float*)preProcframe.data);

                return true;
            }
        }
        else {
            vcap.release();
            return false;// Manually stopped the application so returning false to stop input.
        }
    }

    // Input callback function to fetch frames and preprocess them
    bool outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int streamLabel) {

        //Ouput from the post-processing model is a vector of size 1
        //So copying only the first featuremap
        src[0]->get_data(mxa_output);
        // cv::Mat inImage;
        {
            std::lock_guard<std::mutex> ilock(frame_queue_mutex);
            // pop from frame queue
            displayImage = frames_queue.front();
            frames_queue.pop_front();
        }// releases in frame queue lock

        //Get the detections from model output
        std::vector<detectedObj> detected_objectVector = get_detections(mxa_output, num_boxes);

        // draw boundign boxes
        draw_bounding_box(displayImage, detected_objectVector);

        cv::Mat bgrDisplay;
        cv::cvtColor(displayImage, bgrDisplay, cv::COLOR_RGB2BGR);

        cv::imshow(window_name, bgrDisplay);

        if (cv::waitKey(1) == 'q') {
            runflag.store(false);
        }
    }

public:
    //YoloV26(MX::Runtime::MxAccl* accl, std::string video_src, int index) {
    YoloV26(MX::Runtime::MxAccl* accl, std::string video_src, int index)
        : window_name("Live Feed - Camera " + std::to_string(index)) {
           
            video_src = video_src;
            // If the input is a camera, try to use optimal settings
            if (video_src.substr(0, 3) == "cam") {
                int device = std::stoi(video_src.substr(4));
                src_is_cam = true;
               
                if (!openCamera(vcap, device, cv::CAP_ANY)) {
                    throw(std::runtime_error("Failed to open: " + video_src));
                }
            }
            else if (video_src.substr(0, 3) == "vid") {  
                vcap.open(video_src.substr(4, size(video_src)), cv::CAP_ANY);
                src_is_cam = false;
            }
            
            else {
                throw(std::runtime_error("Given video src: " + video_src + " is invalid" +
                    "\n\n\tUse ./objectDetection cam:<camera index>,vid:<path to video file>,cam:<camera index>,vid:<path to video file>\n\n"));
            }
            if (!vcap.isOpened()) {
                std::cout << "videocapture for " << video_src << " is NOT opened, \n try giving full absolete paths for video files and correct camera index for cmameras \n";
                runflag.store(false);
            }
           
            // Getting input image dimensions
            input_image_width = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_WIDTH));
            input_image_height = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_HEIGHT));

            model_info = accl->get_model_info(0);//Getting model info of 0th model which is the only model in this DFP
            mxa_output = new float[num_boxes * 6];//Creating the memory of output (max_boxes X num_box_parameters) - YOLOv26 has 6 params per box 

            //Getting model input shapes and display size
            model_input_height = model_info.in_featuremap_shapes[0][0];
            model_input_width = model_info.in_featuremap_shapes[0][1];

            //Connecting the stream to the accl object. As the callback functions are defined as part of the class
            //YoloV26 we should bind them with the possible input parameters
            auto in_cb = std::bind(&YoloV26::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
            auto out_cb = std::bind(&YoloV26::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);
            accl->connect_stream(in_cb, out_cb, index/**Unique Stream Idx */, 0/**Model Idx */);

            //Starts the callbacks when the call is started
            runflag.store(true);
        }
        ~YoloV26() {
            delete[] mxa_output;
            mxa_output = NULL;

            vcap.release();
            //cv::destroyAllWindows();
            cv::destroyWindow(window_name);
           
        }
};
    


std::vector<string> getAvailableCameraIndices(int max_index = 50) {
    std::vector<string> available;
    for (int i = 0; i < max_index; ++i) {
        cv::VideoCapture cap(i, cv::CAP_ANY);  // or CAP_MSMF, CAP_DSHOW
        if (cap.isOpened()) {
            available.push_back("cam:" + to_string(i));
            cap.release();
        }
    }
    return available;
}

int main(int argc, char* argv[]) {

    std::cout << "main function \n";

    signal(SIGINT, signal_handler);
    vector<string> video_src_list;

    std::string video_str = "cam:0";

    // Iterate through the arguments
    for (int i = 1; i < argc; i++) {

        std::string arg = argv[i];

        // Handle -d or --dfp_path
        if (arg == "-d" || arg == "--dfp_path") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                model_path = argv[++i];
            }
            else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle -m or --post_model
        else if (arg == "-m" || arg == "--post_model") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                postprocessing_model_path = argv[++i];
            }
            else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle --video_paths
        else if (arg == "--video_paths") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                video_str = argv[++i];
                size_t pos = 0;
                std::string token;
                std::string delimiter = ",";
                while ((pos = video_str.find(delimiter)) != std::string::npos) {
                    token = video_str.substr(0, pos);
                    video_src_list.push_back(token);
                    video_str.erase(0, pos + delimiter.length());
                }
                video_src_list.push_back(video_str);
            }
            else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle unknown options
        else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // if video_paths arg isn't passed - use default video string.
    if (video_src_list.size() == 0) {

        video_src_list.push_back(video_str);
    }



    MX::Runtime::MxAccl accl{
                fs::path(model_path),                      // DFP path
                std::vector<int>{0},                    // device_ids_to_use
                std::array<bool, 2>{true, true},        // use_model_shape
                false,                                  // local_mode
                MX::RPC::SchedulerOptions{600, 0, false, 16, 12},  // sched_options
                MX::RPC::ClientOptions{false, 0},       // client_options
                "localhost",                            // server_addr
                10000,                                  // server_port_base
                false };


    // Connecting the post-processing model obtained from the autocrop of neural compiler to get the final output.
    // The second parameter is required as the output shape of this particular post-processing model is variable
    // and accl requires to know maximum possible size of the output. In this case it is (max_possible_boxes * size_of_box = 300 * 6 = 1800).
    accl.connect_post_model(fs::path(postprocessing_model_path), 0, std::vector<size_t>{300 * 6});


   //Creating a YoloV26 object for each stream which also connects the corresponding stream to accl.
    int free_cam_idx = 0;
    std::vector<YoloV26*>yolo_objs;
    for (int i = 0; i < video_src_list.size(); ++i) {
        if (video_src_list[i].substr(0, 3) == "cam") {
            video_src_list[i] = free_cams[free_cam_idx];
            free_cam_idx++;
        }
        
        YoloV26* obj = new YoloV26(&accl, video_src_list[i], i);
        yolo_objs.push_back(obj);

    }


    //Run the accelerator and wait
    accl.start();
    accl.wait();
    accl.stop();


    //Cleanup
    for(int i =0; i<video_src_list.size();++i ){
        delete yolo_objs[i];
    }
}
