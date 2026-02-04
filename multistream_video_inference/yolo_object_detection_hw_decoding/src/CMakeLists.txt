cmake_minimum_required(VERSION 3.10)

get_filename_component(app_name main NAME)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-but-set-variable -Wno-narrowing -Wno-unused-result -Wno-deprecated-declarations -Wno-sign-compare -Wno-literal-suffix -Werror")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Og -g -fopenmp -fsanitize=address -fsanitize=leak")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Og -g -fopenmp -fsanitize=address -fsanitize=leak")
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -g -ggdb -fopenmp -march=native -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g -ggdb -fopenmp -march=native -fno-omit-frame-pointer")
endif()

find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)
find_package(CURL REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWSCALE REQUIRED libswscale)

# Include directories for FFmpeg and OpenCV
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB local_src
    "*.c"
    "*.cpp"
    "*.h"
	)

add_executable(${app_name} ${local_src})

target_link_libraries(${app_name}  mx_accl  ${OpenCV_LIBS} mxutils_gui)
target_link_libraries(${app_name} Qt5::Widgets Qt5::Core Qt5::Gui)

# Link against the FFmpeg libraries, OpenCV, and any extra system libraries like dl and pthread.
target_link_libraries(${app_name}
    ${OpenCV_LIBS}
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    ${CURL_LIBRARIES}
    dl
    pthread
)