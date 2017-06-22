# project
cmake_minimum_required (VERSION 2.6)
project (opticalFlow)

# directories
set (OPTFLOW_SRC_DIR src)

SET(OpenCV_DIR C:/Users/megamusz/Downloads/opencv-3.2.0-vc14/opencv/build/)
# include directory
include_directories("${OPTFLOW_SRC_DIR}")

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# add viso2 include
include_directories( "../libviso2/src")
link_directories ("../../libviso2/build/Release")

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB OPTFLOW_SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)

# demo program
add_executable(optflow ${OPTFLOW_SRC_FILES})
target_link_libraries (optflow ${OpenCV_LIBS} libviso2.lib)

