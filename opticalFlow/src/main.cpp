#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <fstream>
#include <viso_stereo.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/optflow.hpp"

using namespace cv::ximgproc;
using namespace cv::optflow;
using namespace std;
using namespace cv;

//get the project matrix from KITTI calibration file, as well as initialize the param for VOstereo
Matrix getProjectionMatrix(string calibFileName, VisualOdometryStereo::parameters& param) {
	ifstream f(calibFileName);
	double *pDat = new double[4 * 3];
	//string temp;
	f >> string();
	
	for (int m = 0; m < 3; m++) {
		for (int n = 0; n < 4; n++) {
			f >> pDat[m * 4 + n];
			//cout << pDat[m * 4 + n];
		}
	}
	f >> string();
	double baseFocal;
	for (int i = 0; i < 4; i++)
		f >> baseFocal;

	Matrix P(3, 4, pDat);

	param.calib.f = pDat[0];	// focal length in pixels
	param.calib.cu = pDat[2];	// principal point (u-coordinate) in pixels
	param.calib.cv = pDat[6];	// principal point (v-coordinate) in pixels
	param.base = -baseFocal/param.calib.f; // baseline in meters, = P1(1, 4)/f

	delete[] pDat;

	return P;
}
typedef struct {
	ushort valid;
	ushort mvy;
	ushort mvx;
} FlowPix;

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
void writeFalseColor(Mat& flow, const char* fileName, float max_flow = 128)
{
	float n = 8; // multiplier
	int m_height = flow.rows;
	int m_width = flow.cols;
	Mat image(m_height, m_width, CV_8UC3);
	for (int32_t y = 0; y<m_height; y++) {
		for (int32_t x = 0; x<m_width; x++) {
			float r = 0, g = 0, b = 0;
			float mvx = flow.at<Vec2f>(y, x).val[0];
			float mvy = flow.at<Vec2f>(y, x).val[1];
			//cout << mvx << mvy << endl;
			float mag = sqrt(mvx*mvx + mvy*mvy);
			float dir = atan2(mvy, mvx);
			float h = fmod(dir / (2.0*3.1415926) + 1.0, 1.0);
			float s = std::min(std::max(mag*n / max_flow, 0.0f), 1.0f);
			float v = std::min(std::max(n - s, 0.0f), 1.0f);
			hsvToRgb(h, s, v, r, g, b);
			//cout << r <<", "<< g <<", "<< b << endl;
			image.at<Vec3b>(y, x) = Vec3b(b*255.0f, g*255.0f, r*255.0f);
		}
	}

	imwrite(fileName, image);

}


int main(int argc, char** argv)
{
	if (argc < 3) {
		cout << "usage: optflow directory seqenceIdx" << endl;
	}
	//C:\Users\megamusz\Desktop\data_stereo_flow\training
	string dir = argv[1];
	int seqIdx = atoi(argv[2]);


	// for a full parameter list, look at: viso_stereo.h
	VisualOdometryStereo::parameters param;

	// current pose (this matrix transforms a point from the current
	// frame's camera coordinates to the first frame's camera coordinates)
	Matrix pose = Matrix::eye(4);
	char calibFileName[256];
	sprintf(calibFileName, "/calib/%06d.txt", seqIdx);
	Matrix P = getProjectionMatrix(dir + calibFileName, param);
	// init visual odometry
	VisualOdometryStereo viso(param);

	//cout << P << endl;
	// loop through all frame 10 & 11
	Mat disp;
	int width, height;
	uint8_t* left_img_data = NULL;
	uint8_t* right_img_data = NULL;
	uchar* Ipred = NULL; 
	Mat It;
	for (int32_t i = 0; i < 2; i++) {
		// input file names
		int frameIdx = i == 0 ? 10 : 11;
		char base_name[256]; sprintf(base_name, "%06d_%d.png", seqIdx, frameIdx);
		string left_img_file_name	= dir + "/colored_0/" + base_name;
		string right_img_file_name	= dir + "/colored_1/" + base_name;

		
		// catch image read/write errors here
		try {
			cout << left_img_file_name << endl;
			Mat left_img = imread(left_img_file_name, IMREAD_GRAYSCALE);
			if (i == 0)
				It = left_img;
			Mat right_img = imread(right_img_file_name, IMREAD_GRAYSCALE);
			//estimate the disparity map
			if (i == 0) {
				int max_disp = 128;
#if 0 
				Mat left_for_matcher, right_for_matcher;
				Mat left_disp, right_disp;
				Mat filtered_disp;
				
				int wsize = 7;
				max_disp /= 2;
				if (max_disp % 16 != 0)
					max_disp += 16 - (max_disp % 16);
				resize(left_img, left_for_matcher, Size(), 0.5, 0.5);
				resize(right_img, right_for_matcher, Size(), 0.5, 0.5);

				Ptr<StereoBM> left_matcher = StereoBM::create(max_disp, wsize);
				Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilter(left_matcher);
				Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

				//cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
				//cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

				double matching_time = (double)getTickCount();
				left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
				right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
				matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();

				double lambda = 10;
				double sigma = 5;
				wls_filter->setLambda(lambda);
				wls_filter->setSigmaColor(sigma);
	
				wls_filter->filter(left_disp, left_img, disp, right_disp);
				Mat disp8;
				disp.convertTo(disp8, CV_8U, 1/16.0);
				imwrite("disp.png", disp8);
#else
				int sgbmWinSize = 7;

				Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, max_disp, sgbmWinSize);

				sgbm->setP1(8 * sgbmWinSize*sgbmWinSize);
				sgbm->setP2(32 * sgbmWinSize*sgbmWinSize);
				sgbm->setMinDisparity(0);
				sgbm->setNumDisparities(max_disp);
				sgbm->setUniquenessRatio(0);
				sgbm->setSpeckleWindowSize(0);
				sgbm->setSpeckleRange(32);
				sgbm->setDisp12MaxDiff(1000);
				sgbm->setMode(StereoSGBM::MODE_HH);

				sgbm->compute(left_img, right_img, disp);
#endif
			}
			// image dimensions
			width = left_img.cols;
			height = left_img.rows;

			// convert input images to uint8_t buffer
			if (left_img_data == NULL || right_img_data == NULL || Ipred == NULL) {
				left_img_data = (uint8_t*)malloc(width*height * sizeof(uint8_t));
				right_img_data = (uint8_t*)malloc(width*height * sizeof(uint8_t));
				Ipred = new uchar[width*height];
			}
			int32_t k = 0;
			for (int32_t v = 0; v<height; v++) {
				uchar* leftPtr = left_img.ptr<uchar>(v);
				uchar* rightPtr = right_img.ptr<uchar>(v);
				for (int32_t u = 0; u<width; u++) {
					left_img_data[k] = leftPtr[u];
					right_img_data[k] = rightPtr[u];
					if (i == 0)
						Ipred[k] = left_img_data[k];
					k++;
				}
			}

			// status
			cout << "Processing: Frame: " << frameIdx << endl;
			// compute visual odometry
			int32_t dims[] = { width,height,width };
			if (viso.process(left_img_data, right_img_data, dims)) {

				// on success, update current pose
				pose = pose * Matrix::inv(viso.getMotion());

				// output some statistics
				double num_matches = viso.getNumberOfMatches();
				double num_inliers = viso.getNumberOfInliers();
				cout << ", Matches: " << num_matches;
				cout << ", Inliers: " << 100.0*num_inliers / num_matches << " %" << ", Current pose: " << endl;
				cout << pose << endl << endl;

			}
			else {
				cout << " ... failed!" << endl;
			}


			// catch image read errors here
		}
		catch (...) {
			cerr << "ERROR: Couldn't read input files!" << endl;
			return 1;
		}
	}


	//get the predicted flow
	Matrix Xt(4, 1);
	double qDat[4][4] = { 0 };
	qDat[0][0] = qDat[1][1] = 1;
	qDat[0][3] = -param.calib.cu;
	qDat[1][3] = -param.calib.cv;
	qDat[2][3] = param.calib.f;
	qDat[3][2] = -1 / param.base;
	Matrix Q(4, 4, &qDat[0][0]);//triangular matrix

	//cout << Q << endl;
	double w;

	FlowPix* flowDat = new FlowPix[height*width];
	memset(flowDat, 0, sizeof(FlowPix)*width*height);
	
	double xtVal[4];
	
	//memset(Ipred, 0, sizeof(uchar)*width*height);
	memcpy(Ipred, left_img_data, sizeof(uchar)*width*height);
	uchar* hasValidPixel = new uchar[width*height];
	memset(hasValidPixel, 0, sizeof(uchar)*width*height);

	for (int j = 0; j < height; j++) {
		short * ptrDisp = disp.ptr<short>(j);
		
		for (int i = 0; i < width; i++) {
			double dis = ptrDisp[i] / 16.0;
			
			if (dis < 0) {
				
				dis = ptrDisp[128] / 16.0;
				if(dis<0)
					continue;
			}

			xtVal[0] = i; xtVal[1] = j; xtVal[2] = -dis; xtVal[3] = 1;
			Matrix Xt(4, 1, xtVal);
			Matrix recon = Q*Xt;
			Matrix Xt1 = viso.getMotion() * recon;

			Matrix est = P*Xt1;
			est.getData(&w, 2, 0, 2, 0);
			est = est / w;

			if (w == 0)
				continue;

			double xt1, yt1;
			est.getData(&xt1, 0, 0, 0, 0);
			est.getData(&yt1, 1, 0, 1, 0);

			flowDat[j*width + i].valid = 1;
			flowDat[j*width + i].mvx = 64.0 * (xt1 - i) + 32768;
			flowDat[j*width + i].mvy = 64.0 * (yt1 - j) + 32768;

			int targetX = xt1 + 0.5;
			int targetY = yt1 + 0.5;
			if (targetX >= 0 && targetX < width && targetY >= 0 && targetY < height) {//get predicted pixel
				Ipred[j*width + i] = left_img_data[targetY*width + targetX];
				hasValidPixel[j*width + i] = 255;
			}
		}
	}


	char predImageName[256];
	sprintf(predImageName, "%06d_10_pred.png", seqIdx);
	Mat Ip(height, width, CV_8U, Ipred);


	Mat flow(height, width, CV_16UC3, flowDat);
	char flowName[256];
	sprintf(flowName, "%06d_10.png", seqIdx);
	imwrite(flowName, flow);

	

#if 1 
	//correction stage
	Mat flowC;
	//calcOpticalFlowSparseToDense(It, Ip, flowC, 4, 128, 0.001);
	calcOpticalFlowFarneback(It, Ip, flowC, 0.8, 1, 11, 1, 2, 1.5, 0);
	writeFalseColor(flowC, "flowC.png", 10);
	//u(x, y) = ?u(x, y) + upred(x + ?u(x, y), y + ?v(x, y))
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			Vec2f deltaFlow = flowC.at<Vec2f>(j, i);
			if (hasValidPixel[j*width + i] > 0) {// && (abs(deltaFlow.val[0]) > 5 || abs(deltaFlow.val[1]) > 5)) { //have valid wrap pixel from It+1
				int tx = i;
				int ty = j;
				//int tx = std::max(std::min(width - 1, int(i + deltaFlow.val[0] + 0.5)), 0);
				//int ty = std::max(std::min(height - 1, int(j + deltaFlow.val[1] + 0.5)), 0);
				
				float mvx = (int(flowDat[ty*width + tx].mvx) - 32768) / 64.0;
				float mvy = (int(flowDat[ty*width + tx].mvy) - 32768) / 64.0;
				

				flowDat[j*width + i].mvx = 64.0*(mvx + deltaFlow.val[0]) + 32768;
				flowDat[j*width + i].mvy = 64.0*(mvy + deltaFlow.val[1]) + 32768;
			}
		}
	}
#endif
	imwrite(predImageName, Ip);
	imwrite("It.png", It);

	sprintf(flowName, "%06d_10_c.png", seqIdx);
	imwrite(flowName, flow);

	delete[] flowDat;
	// release uint8_t buffers
	free(left_img_data);
	free(right_img_data);
	delete[] Ipred;

	return 0;
}