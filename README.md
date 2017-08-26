# stereoVO-opticalflow
a optical flow method that using stereo VO, similar with "A Prediction-Correction Approach for Real-Time Optical Flow Computation Using Stereo"

Pre-request:
1. OpenCV 3.2
2. libviso2, could be found in http://www.cvlibs.net/software/libviso/ 

Step 1.
Build libviso2 
the original libviso2 is based on png++ library, the updated version is based on OpenCV for these image IO. Need to first modify the CMakeLists.txt to set OpenCV_DIR to your path. 
the default build is release version. for debug need to update the CMakeLists.txt.

Step 2.
Build opticalFlow
first modify the OpenCV_DIR in CMakeLists.txt to your own, and then generate the solution and build
the default build is release version. for debug need to update the CMakeLists.txt. also need pre-build debug version of libviso2

