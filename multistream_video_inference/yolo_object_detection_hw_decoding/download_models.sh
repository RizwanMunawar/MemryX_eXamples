#!/bin/bash

# Base URL for all model downloads
BASE_URL="https://developer.memryx.com/model_explorer/2p2"

# Model URL paths - Fill in the custom path portion after the base URL for each model
# Each URL should point to a zip file containing both model.dfp and post.onnx
declare -A MODEL_PATHS

# YOLOv8 models
MODEL_PATHS["8n640"]="YOLO_v8_nano_640_640_3_onnx.zip"
MODEL_PATHS["8s640"]="YOLO_v8_small_640_640_3_onnx.zip"
MODEL_PATHS["8m640"]="YOLO_v8_medium_640_640_3_onnx.zip"

# YOLOv9 models
MODEL_PATHS["9t640"]="YOLO_v9_tiny_640_640_3_onnx.zip"
MODEL_PATHS["9s640"]="YOLO_v9_small_640_640_3_onnx.zip"
MODEL_PATHS["9m640"]="YOLO_v9_medium_640_640_3_onnx.zip"

# YOLOv10 models
MODEL_PATHS["10n320"]="YOLO_v10_nano_320_320_3_onnx.zip"
MODEL_PATHS["10n480"]="YOLO_v10_nano_480_480_3_onnx.zip"
MODEL_PATHS["10s320"]="YOLO_v10_small_320_320_3_onnx.zip"
MODEL_PATHS["10s480"]="YOLO_v10_small_480_480_3_onnx.zip"
MODEL_PATHS["10m320"]="YOLO_v10_medium_320_320_3_onnx.zip"
MODEL_PATHS["10m480"]="YOLO_v10_medium_480_480_3_onnx.zip"

# YOLOv11 models
MODEL_PATHS["11n320"]="YOLO11_nano_320_320_3_onnx.zip"
MODEL_PATHS["11n640-opt"]="YOLO11_nano_MXA_Optimized_640_640_3_onnx.zip"
MODEL_PATHS["11n800-opt"]="YOLO11_nano_MXA_Optimized_800_800_3_onnx.zip"
MODEL_PATHS["11s320"]="YOLO11_small_320_320_3_onnx.zip"
MODEL_PATHS["11m320"]="YOLO11_medium_320_320_3_onnx.zip"

# Array of all supported models
ALL_MODELS=(
    "8n640" "8s640" "8m640"
    "9t640" "9s640" "9m640"
    "10n320" "10n480" "10s320" "10s480" "10m320" "10m480"
    "11n320" "11n640-opt" "11n800-opt" "11s320" "11m320"
)

# Function to download a single model
download_model() {
    local model=$1
    local model_dir="assets/models/${model}"
    local temp_zip="${model}.zip"
    
    echo "Downloading ${model}..."
    
    # Check if path is configured
    if [[ "${MODEL_PATHS[$model]}" == "PASTE_PATH_HERE" ]]; then
        echo "Error: Path not configured for ${model}. Please update the script with actual download path."
        return 1
    fi
    
    # Construct full URL
    local full_url="${BASE_URL}/${MODEL_PATHS[$model]}"
    
    # Create directory if it doesn't exist
    mkdir -p "${model_dir}"
    
    # Download zip file
    wget -O "${temp_zip}" "${full_url}"
    if [ $? -ne 0 ]; then
        echo "Error downloading ${model} zip file"
        rm -f "${temp_zip}"
        return 1
    fi
    
    # Extract zip file to model directory
    unzip -o "${temp_zip}" -d "${model_dir}"
    if [ $? -ne 0 ]; then
        echo "Error extracting ${model} zip file"
        rm -f "${temp_zip}"
        return 1
    fi
    
    # Clean up zip file
    rm -f "${temp_zip}"
    
    # Find and rename .dfp file to model.dfp
    dfp_file=$(find "${model_dir}" -type f -name "*.dfp" | head -n 1)
    if [[ -n "$dfp_file" ]] && [[ "$dfp_file" != "${model_dir}/model.dfp" ]]; then
        mv "$dfp_file" "${model_dir}/model.dfp"
    fi
    
    # Find and rename .onnx file to post.onnx
    onnx_file=$(find "${model_dir}" -type f -name "*.onnx" | head -n 1)
    if [[ -n "$onnx_file" ]] && [[ "$onnx_file" != "${model_dir}/post.onnx" ]]; then
        mv "$onnx_file" "${model_dir}/post.onnx"
    fi
    
    # Verify both files exist
    if [[ ! -f "${model_dir}/model.dfp" ]] || [[ ! -f "${model_dir}/post.onnx" ]]; then
        echo "Warning: Expected files (model.dfp and/or post.onnx) not found after extraction for ${model}"
        return 1
    fi
    
    echo "Successfully downloaded and extracted ${model}"
    return 0
}

# Function to check if model name is valid
is_valid_model() {
    local model=$1
    for valid_model in "${ALL_MODELS[@]}"; do
        if [ "$valid_model" == "$model" ]; then
            return 0
        fi
    done
    return 1
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] [MODEL_NAMES...]"
    echo ""
    echo "Download YOLO model files for MemryX accelerators"
    echo ""
    echo "Options:"
    echo "  --all           Download all 21 supported models"
    echo "  -h, --help      Display this help message"
    echo ""
    echo "Supported Models:"
    echo "  YOLOv8: 8n640, 8s640, 8m640"
    echo "  YOLOv9: 9t640, 9s640, 9m640"
    echo "  YOLOv10: 10n320, 10n480, 10s320, 10s480, 10m320, 10m480"
    echo "  YOLOv11: 11n320, 11n480-opt, 11n640-opt, 11n800-opt,"
    echo "           11s320, 11s480-opt, 11s640-opt, 11m320"
    echo ""
    echo "Examples:"
    echo "  $0 8s640                     # Download a single model"
    echo "  $0 --all                     # Download all models"
    echo "  $0 11n640-opt 11s480-opt     # Download multiple models"
}

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided"
    usage
    exit 1
fi

MODELS_TO_DOWNLOAD=()

# Process arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            MODELS_TO_DOWNLOAD=("${ALL_MODELS[@]}")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
        *)
            if is_valid_model "$1"; then
                MODELS_TO_DOWNLOAD+=("$1")
            else
                echo "Error: Invalid model name '$1'"
                echo "Run '$0 --help' to see supported models"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if we have models to download
if [ ${#MODELS_TO_DOWNLOAD[@]} -eq 0 ]; then
    echo "Error: No valid models specified"
    usage
    exit 1
fi

# Create base directory
mkdir -p assets/models

# Download models
echo "Downloading ${#MODELS_TO_DOWNLOAD[@]} model(s)..."
echo ""

FAILED_MODELS=()
SUCCESS_COUNT=0

for model in "${MODELS_TO_DOWNLOAD[@]}"; do
    if download_model "$model"; then
        ((SUCCESS_COUNT++))
    else
        FAILED_MODELS+=("$model")
    fi
    echo ""
done

# Summary
echo "======================================"
echo "Download Summary:"
echo "======================================"
echo "Successfully downloaded: ${SUCCESS_COUNT}/${#MODELS_TO_DOWNLOAD[@]} models"

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed models: ${FAILED_MODELS[*]}"
    exit 1
else
    echo "All models downloaded successfully!"
    exit 0
fi
