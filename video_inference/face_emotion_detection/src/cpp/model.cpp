#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;  // Alias for filesystem

#include "model.h"

// Constructor for the MPFaceDetector class
MPFaceDetector::MPFaceDetector() {
    loadAnchors("../anchors.txt");  // Load anchor points from file
    initializeConstants();          // Initialize constants used in the model
}

// loadAnchors method to load anchor points from a file
void MPFaceDetector::loadAnchors(const std::string& filename) {
    std::ifstream file(filename);  // Open the anchor file
    if (!file.is_open()) {         // Check if the file was opened successfully
        std::cerr << "Failed to open anchor file: " << filename << std::endl;
        return;
    }

    std::vector<float> anchorValues;  // Store the anchor values
    std::string line;
    while (std::getline(file, line)) {  // Read each line in the file
        std::replace(line.begin(), line.end(), ',', ' ');  // Replace commas with spaces
        std::istringstream iss(line);
        float x1, y1, x2, y2;
        if (!(iss >> x1 >> y1 >> x2 >> y2)) { break; }  // Parse coordinates from line

        // Store anchor values (x1, y1, x2, y2)
        anchorValues.push_back(x1);
        anchorValues.push_back(y1);
        anchorValues.push_back(x2);
        anchorValues.push_back(y2);
    }
    file.close();  // Close the file

    // Convert the vector of anchor values to a torch::Tensor
    int64_t numAnchors = anchorValues.size() / 4;  // Calculate the number of anchors
    anchors = torch::from_blob(anchorValues.data(), {numAnchors, 4}, torch::kFloat).clone();  // Create tensor
}

// Initialize constants for model configuration
void MPFaceDetector::initializeConstants() {
    outputMap = {  // Map output layer names to indices
        {"regressor_8", 2},
        {"regressor_16", 3},
        {"classificator_8", 0},
        {"classificator_16", 1},
    };
    // Set scaling factors and score threshold based on model version
    if (backModel) {
        xScale = yScale = hScale = wScale = 256.0f;
        minScoreThresh = 0.65f;
    } else {
        xScale = yScale = hScale = wScale = 128.0f;
        minScoreThresh = 0.75f;
    }
}

// Decode bounding box predictions using anchor points
torch::Tensor MPFaceDetector::_decode_boxes(const torch::Tensor& raw_boxes, const torch::Tensor& anchors) {
    auto boxes = torch::zeros_like(raw_boxes);  // Initialize tensor for decoded boxes

    // Calculate box center and size using anchor points
    auto x_center = raw_boxes.index({"...", 0}) / xScale * anchors.index({"...", 2}) + anchors.index({"...", 0});
    auto y_center = raw_boxes.index({"...", 1}) / yScale * anchors.index({"...", 3}) + anchors.index({"...", 1});
    auto w = raw_boxes.index({"...", 2}) / wScale * anchors.index({"...", 2});
    auto h = raw_boxes.index({"...", 3}) / hScale * anchors.index({"...", 3});

    // Set box coordinates (ymin, xmin, ymax, xmax)
    boxes.index_put_({"...", 0}, y_center - h / 2.0);  // ymin
    boxes.index_put_({"...", 1}, x_center - w / 2.0);  // xmin
    boxes.index_put_({"...", 2}, y_center + h / 2.0);  // ymax
    boxes.index_put_({"...", 3}, x_center + w / 2.0);  // xmax

    // Decode keypoints (facial landmarks)
    for (int k = 0; k < 6; ++k) {
        int offset = 4 + k * 2;
        auto keypoint_x = raw_boxes.index({"...", offset}) / xScale * anchors.index({"...", 2}) + anchors.index({"...", 0});
        auto keypoint_y = raw_boxes.index({"...", offset + 1}) / yScale * anchors.index({"...", 3}) + anchors.index({"...", 1});
        boxes.index_put_({"...", offset}, keypoint_x);  // Keypoint x
        boxes.index_put_({"...", offset + 1}, keypoint_y);  // Keypoint y
    }

    return boxes;  // Return decoded boxes and keypoints
}

// Convert raw tensors (bounding boxes and scores) to detections
std::vector<torch::Tensor> MPFaceDetector::_tensors_to_detections(const torch::Tensor& raw_box_tensor, const torch::Tensor& raw_score_tensor, const torch::Tensor& anchors) {
    auto detection_boxes = _decode_boxes(raw_box_tensor, anchors);  // Decode bounding boxes

    auto thresh = scoreClippingThresh;  // Set score clipping threshold
    auto raw_scores = raw_score_tensor.clamp(-thresh, thresh);  // Clip scores to threshold

    auto detection_scores = raw_score_tensor.sigmoid().squeeze(-1);  // Apply sigmoid to get confidence scores

    std::vector<torch::Tensor> output_detections;
    auto mask = detection_scores >= minScoreThresh;  // Create mask for confident detections

    // Process each image in the batch
    for (int i = 0; i < raw_box_tensor.size(0); ++i) {
        auto current_mask = mask[i];  // Apply mask to filter boxes and scores
        auto boxes = detection_boxes[i].index({current_mask});
        auto scores = detection_scores[i].index({current_mask}).unsqueeze(-1);

        if (boxes.dim() == 1) {  // Handle case where only one detection is present
            boxes = boxes.unsqueeze(0);
        }

        // Concatenate boxes and scores into final detections
        auto detections = torch::cat({boxes, scores}, 1); 
        output_detections.push_back(detections);  // Store the result
    }

    return output_detections;  // Return detections for each image
}

// Calculate intersection area between two sets of boxes (used for IoU calculation)
torch::Tensor MPFaceDetector::intersect(const torch::Tensor& box_a, const torch::Tensor& box_b) {
    auto A = box_a.size(0);  // Number of boxes in set A
    auto B = box_b.size(0);  // Number of boxes in set B

    // Calculate the coordinates of the intersection box
    auto max_xy = torch::min(box_a.slice(1, 2, torch::indexing::None).unsqueeze(1).expand({A, B, 2}),
                            box_b.slice(1, 2, torch::indexing::None).unsqueeze(0).expand({A, B, 2}));

    auto min_xy = torch::max(box_a.slice(1, 0, 2).unsqueeze(1).expand({A, B, 2}),
                            box_b.slice(1, 0, 2).unsqueeze(0).expand({A, B, 2}));

    auto inter = torch::clamp(max_xy - min_xy, 0);  // Clamp to ensure non-negative values

    // Calculate the area of the intersection
    auto inter_area = inter.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) * 
                      inter.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});

    return inter_area;  // Return intersection areas
}

// Compute Jaccard overlap (Intersection over Union, IoU) between two sets of boxes
torch::Tensor MPFaceDetector::jaccard(const torch::Tensor& box_a, const torch::Tensor& box_b) {
    auto inter = MPFaceDetector::intersect(box_a, box_b);  // Compute intersection area

    // Compute area of boxes in set A
    auto width = box_a.index({torch::indexing::Slice(), 2}) - box_a.index({torch::indexing::Slice(), 0});
    auto height = box_a.index({torch::indexing::Slice(), 3}) - box_a.index({torch::indexing::Slice(), 1});
    auto area_a = (width * height).unsqueeze(1).expand_as(inter);

    // Compute area of boxes in set B
    auto width_b = box_b.index({torch::indexing::Slice(), 2}) - box_b.index({torch::indexing::Slice(), 0});
    auto height_b = box_b.index({torch::indexing::Slice(), 3}) - box_b.index({torch::indexing::Slice(), 1});
    auto area_b = (width_b * height_b).unsqueeze(0).expand_as(inter);
    
    // Compute union area
    auto union_ = area_a + area_b - inter;
    return inter / union_;  // Return IoU (intersection over union)
}

// Compute overlap similarity (IoU) between one box and a set of other boxes
torch::Tensor MPFaceDetector::overlap_similarity(const torch::Tensor& box, const torch::Tensor& other_boxes) {
    return MPFaceDetector::jaccard(box.unsqueeze(0), other_boxes).squeeze(0);  // Compute IoU for the first box
}

// Perform weighted Non-Maximum Suppression (NMS) on detections
std::vector<torch::Tensor> MPFaceDetector::_weighted_non_max_suppression(const torch::Tensor& detections) {
    if (detections.size(0) == 0) return {};  // Return if there are no detections

    std::vector<torch::Tensor> output_det;  // Store final detections

    // Sort detections by confidence scores in descending order
    torch::Tensor column_values = detections.index({torch::indexing::Slice(), 16});
    torch::Tensor remaining = torch::argsort(column_values, 0, true);

    while (remaining.numel() > 0) {  // Loop until no boxes remain
        auto detection = detections.index({remaining[0]});  // Get the highest scoring detection
        auto first_box = detection.index({torch::indexing::Slice(0, 4)});  // Get the bounding box

        auto other_boxes = detections.index({remaining, torch::indexing::Slice(0, 4)});  // Get all boxes
        auto ious = MPFaceDetector::overlap_similarity(first_box, other_boxes);  // Compute IoU

        auto mask = ious > minSuppressionThreshold;  // Apply IoU threshold
        auto overlapping = remaining.index({mask});  // Get overlapping boxes
        auto i_m = torch::logical_not(mask);
        remaining = remaining.index({i_m});  // Remove processed boxes

        auto weighted_detection = detection.clone();  // Create a copy of the detection

        // Perform weighted averaging if multiple overlapping detections exist
        if (overlapping.size(0) > 1) {
            auto coordinates = detections.index_select(0, overlapping).slice(1, 0, 17);
            auto scores = detections.index_select(0, overlapping).slice(1, 16, 17);
            auto total_score = scores.sum();

            auto weighted = (coordinates * scores).sum(0) / total_score;
            weighted_detection.slice(0, 0, 17).copy_(weighted);  // Update detection with weighted values
            auto total_score_ratio = total_score / overlapping.size(0);
            weighted_detection.index_put_({16}, total_score_ratio);  // Update score
        }

        output_det.push_back(weighted_detection);  // Add weighted detection to output
    }

    return output_det;  // Return filtered detections
}

// Postprocess the output feature maps and return the final detections
torch::Tensor MPFaceDetector::postprocess(const std::vector<float*>& ofmaps) {
    // Load classification and regression output feature maps
    auto cls8 = torch::from_blob(ofmaps[outputMap["classificator_8"]], {512, 1}).clone();
    auto cls16 = torch::from_blob(ofmaps[outputMap["classificator_16"]], {384, 1}).clone();
    auto reg8 = torch::from_blob(ofmaps[outputMap["regressor_8"]], {512, 16}).clone();
    auto reg16 = torch::from_blob(ofmaps[outputMap["regressor_16"]], {384, 16}).clone();

    // Concatenate classification and regression results
    auto cls = torch::cat({cls8, cls16}, 0).unsqueeze(0);
    auto reg = torch::cat({reg8, reg16}, 0).unsqueeze(0);

    // Convert raw tensors to detections
    auto detections = MPFaceDetector::_tensors_to_detections(reg, cls, anchors);

    std::vector<torch::Tensor> filtered_detections;  // Store final detections

    // Perform non-max suppression on the detections
    for (size_t i = 0; i < detections.size(); ++i) {
        auto faces = MPFaceDetector::_weighted_non_max_suppression(detections[i]);
        auto faces_tensor = faces.empty() ? torch::zeros({0, 17}, torch::kFloat32) : torch::stack(faces);  // Stack faces
        filtered_detections.push_back(faces_tensor);
    }

    assert(filtered_detections.size() == 1);  // Ensure batch size is 1

    // Return detections as a CPU tensor
    torch::Tensor dets = filtered_detections[0].detach().to(torch::kCPU);
    return dets;
}
