cmake_minimum_required(VERSION 3.12)
project(feature_detection)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} src/FeatureDetection.cc)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})