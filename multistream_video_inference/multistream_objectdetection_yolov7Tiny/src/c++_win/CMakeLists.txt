cmake_minimum_required(VERSION 3.10)

get_filename_component(app_name multistream_objectdetection NAME)
project(multistream_objectdetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


file(GLOB local_src
    "*.c"
    "*.cpp"
	)

add_executable(
    ${app_name} ${local_src}  
)

message(STATUS "configuring for windows")
set(OpenCV_DIR "C:/OpenCV/opencv/build/x64/vc16/lib/")  # <- this must point to where OpenCVConfig.cmake is
set(MEMRYX_INCLUDE_DIR "C:/Program Files/MemryX/include")
set(ONNXRUNTIME_DIR "C:/onnxruntime")
 
# === Find OpenCV ===
find_package(OpenCV REQUIRED)

 
# === Include directories ===
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "${MEMRYX_INCLUDE_DIR}"
    "${ONNXRUNTIME_DIR}/include"
) 
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
endif()
 
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
)
 
if (NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Cannot find onnxruntime.lib")
endif()
 
find_library(MXACCL_DYNAMIC
    NAMES memx-accl
    PATHS
        "C:/Program Files/MemryX/lib"
    NO_DEFAULT_PATH
)

if (NOT MXACCL_DYNAMIC)
    message(FATAL_ERROR "Cannot find memx-accl.lib")
endif()
 
find_library(MEMX_LIB
    NAMES udriver
    PATHS
        "C:/Program Files/MemryX/lib"
    NO_DEFAULT_PATH
)

if (NOT MEMX_LIB)
    message(FATAL_ERROR "Cannot find udriver.lib")
endif()


target_link_libraries(${app_name}
    ${MXACCL_DYNAMIC}
    ${MEMX_LIB}
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)

 
file(COPY "../../models/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
