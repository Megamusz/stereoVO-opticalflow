#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include <viso_stereo.h>
#include <fstream>
#include "opencv2/optflow.hpp"
using namespace cv::ximgproc;
using namespace cv::optflow;
using namespace std;
using namespace cv;

inline void hsvToRgb(float h, float s, float v, float &r, float &g, float &b) {
	float c = v*s;
	float h2 = 6.0*h;
	float x = c*(1.0 - fabs(fmod(h2, 2.0) - 1.0));
	if (0 <= h2&&h2<1) { r = c; g = x; b = 0; }
	else if (1 <= h2&&h2<2) { r = x; g = c; b = 0; }
	else if (2 <= h2&&h2<3) { r = 0; g = c; b = x; }
	else if (3 <= h2&&h2<4) { r = 0; g = x; b = c; }
	else if (4 <= h2&&h2<5) { r = x; g = 0; b = c; }
	else if (5 <= h2&&h2 <= 6) { r = c; g = 0; b = x; }
	else if (h2>6) { r = 1; g = 0; b = 0; }
	else if (h2<0) { r = 0; g = 1; b = 0; }
}
void writeFalseColor(Mat& flow, const char* fileName, float max_flow = 128);

Matrix getProjectionMatrix(string calibFileName, VisualOdometryStereo::parameters& param);