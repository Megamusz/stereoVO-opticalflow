#include "utils.h"

#define KITTI2015
void writeFalseColor(Mat& flow, const char* fileName, float max_flow)
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

//get the project matrix from KITTI calibration file, as well as initialize the param for VOstereo
Matrix getProjectionMatrix(string calibFileName, VisualOdometryStereo::parameters& param) {
	ifstream f(calibFileName);
	double *pDat = new double[4 * 3];
	string temp;
	f >> temp;

#ifdef KITTI2015	//get Projection matrix of Camera 2
	while (temp.compare("P_rect_00:"))
		f >> temp;
#endif

	for (int m = 0; m < 3; m++) {
		for (int n = 0; n < 4; n++) {
			f >> pDat[m * 4 + n];
			cout << pDat[m * 4 + n] << ", ";
		}
		cout << endl;
	}
	f >> temp;

#ifdef KITTI2015	//get Projection matrix of Camera 3
	while (temp.compare("P_rect_01:"))
		f >> temp;
#endif
	double baseFocal;
	for (int i = 0; i < 4; i++)
		f >> baseFocal;

	//cout << baseFocal << endl;
	Matrix P(3, 4, pDat);

	param.calib.f = pDat[0];	// focal length in pixels
	param.calib.cu = pDat[2];	// principal point (u-coordinate) in pixels
	param.calib.cv = pDat[6];	// principal point (v-coordinate) in pixels
	param.base = -baseFocal / param.calib.f; // baseline in meters, = P1(1, 4)/f

	delete[] pDat;

	return P;
}

Mat readKITTIFlow(String flowName) {
	Mat raw = imread(flowName.c_str(), IMREAD_UNCHANGED);
	Mat f(raw.size(), CV_32FC2);
	for (int j = 0; j < raw.rows; j++) {
		for (int i = 0; i < raw.cols; i++) {
			f.at<Vec2f>(j, i).val[0] = ((int)raw.at<Vec3w>(j, i).val[2] - 32768) / 64.0;
			f.at<Vec2f>(j, i).val[1] = ((int)raw.at<Vec3w>(j, i).val[1] - 32768) / 64.0;
		}
	}
	return f;
}