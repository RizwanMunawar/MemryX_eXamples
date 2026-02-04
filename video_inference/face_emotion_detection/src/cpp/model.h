#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

class MPFaceDetector {
public:
    MPFaceDetector();

    torch::Tensor postprocess(const std::vector<float*>& ofmaps);

    torch::Tensor _decode_boxes(const torch::Tensor& raw_boxes, const torch::Tensor& anchors);

    std::vector<torch::Tensor> _tensors_to_detections(const torch::Tensor& raw_box_tensor, const torch::Tensor& raw_score_tensor, const torch::Tensor& anchors);

    torch::Tensor intersect(const torch::Tensor& box_a, const torch::Tensor& box_b) ;

    torch::Tensor jaccard(const torch::Tensor& box_a, const torch::Tensor& box_b);

    torch::Tensor overlap_similarity(const torch::Tensor& box, const torch::Tensor& other_boxes);

    std::vector<torch::Tensor> _weighted_non_max_suppression(const torch::Tensor& detections);

private:
    torch::Tensor anchors;
    std::map<std::string, int> outputMap;
    int modelheight = 128;
    int modelwidth = 128;
    int numClasses = 1;
    int numAnchors = 896; 
    int numCoords = 16;
    float scoreClippingThresh = 100.0f;
    bool backModel = false;
    float xScale, yScale, hScale, wScale;
    float minScoreThresh;
    float minSuppressionThreshold = 0.3f;

    void loadAnchors(const std::string& filename);
    void initializeConstants();
};
