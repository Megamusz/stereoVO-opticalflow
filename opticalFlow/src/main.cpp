#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>

#include <viso_stereo.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	//C:\Users\megamusz\Desktop\data_stereo_flow\training
	string dir = argv[1];

	// for a full parameter list, look at: viso_stereo.h
	VisualOdometryStereo::parameters param;

	// calibration parameters for sequence 2010_03_09_drive_0019 
	param.calib.f = 707.0912; // focal length in pixels
	param.calib.cu = 601.8873; // principal point (u-coordinate) in pixels
	param.calib.cv = 183.1104; // principal point (v-coordinate) in pixels
	param.base = 0.5372; // baseline in meters, = P1(1, 4)/f

						 // init visual odometry
	VisualOdometryStereo viso(param);

	// current pose (this matrix transforms a point from the current
	// frame's camera coordinates to the first frame's camera coordinates)
	Matrix pose = Matrix::eye(4);
	Matrix P(3, 4);
	P.setVal(param.calib.f, 0, 0, 0, 0);
	P.setVal(param.calib.f, 1, 1, 1, 1);
	P.setVal(param.calib.cu, 0, 2, 0, 2);
	P.setVal(param.calib.cv, 1, 2, 1, 2);
	P.setVal(1, 2, 2, 2, 2);
	cout << P << endl;
	// loop through all frame 10 & 11
	Mat disp;
	int width, height;
	for (int32_t i = 0; i < 2; i++) {
		// input file names
		int frameIdx = i == 0 ? 10 : 11;
		char base_name[256]; sprintf(base_name, "000000_%d.png", frameIdx);
		string left_img_file_name = dir + "/colored_0/" + base_name;
		string right_img_file_name = dir + "/colored_1/" + base_name;


		// catch image read/write errors here
		try {

			// load left and right input image
			//png::image< png::gray_pixel > left_img(left_img_file_name);
			//png::image< png::gray_pixel > right_img(right_img_file_name);
			cout << left_img_file_name << endl;
			Mat left_img = imread(left_img_file_name, IMREAD_GRAYSCALE);
			Mat right_img = imread(right_img_file_name, IMREAD_GRAYSCALE);
			//estimate the disparity map
			if (i == 0) {
				int sgbmWinSize = 7;
				int numberOfDisparities = 128;
				Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, numberOfDisparities, sgbmWinSize);

				sgbm->setP1(8 * sgbmWinSize*sgbmWinSize);
				sgbm->setP2(32 * sgbmWinSize*sgbmWinSize);
				sgbm->setMinDisparity(0);
				sgbm->setNumDisparities(numberOfDisparities);
				sgbm->setUniquenessRatio(10);
				sgbm->setSpeckleWindowSize(100);
				sgbm->setSpeckleRange(32);
				sgbm->setDisp12MaxDiff(2);
				sgbm->setMode(StereoSGBM::MODE_HH);

				sgbm->compute(left_img, right_img, disp);
			}
			// image dimensions
			width = left_img.cols;// left_img.get_width();
			height = left_img.rows;// left_img.get_height();

								   // convert input images to uint8_t buffer
			uint8_t* left_img_data = (uint8_t*)malloc(width*height * sizeof(uint8_t));
			uint8_t* right_img_data = (uint8_t*)malloc(width*height * sizeof(uint8_t));
			int32_t k = 0;
			for (int32_t v = 0; v<height; v++) {
				for (int32_t u = 0; u<width; u++) {
					left_img_data[k] = left_img.at<uchar>(v, u);// left_img.get_pixel(u, v);
					right_img_data[k] = right_img.at<uchar>(v, u);// right_img.get_pixel(u, v);
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

			// release uint8_t buffers
			free(left_img_data);
			free(right_img_data);

			// catch image read errors here
		}
		catch (...) {
			cerr << "ERROR: Couldn't read input files!" << endl;
			return 1;
		}
	}


	//get the predicted flow
	Matrix Xt(4, 1);
	Matrix Q(4, 4);
	Q.setVal(1, 0, 0, 0, 0);
	Q.setVal(1, 1, 1, 1, 1);
	Q.setVal(-param.calib.cu, 0, 3, 0, 3);
	Q.setVal(-param.calib.cv, 1, 3, 1, 3);
	Q.setVal(param.calib.f, 2, 3, 2, 3);
	Q.setVal(-1 / param.base, 3, 2, 3, 2);
	cout << Q << endl;
	Mat flow(height, width, CV_16UC3);
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			double dis = disp.at<short>(j, i) / 16.0;
			Vec3w mv;
			mv.val[0] = 0; //b
			mv.val[1] = 0; //g
			mv.val[2] = 0; //r
			flow.at<Vec3w>(j, i) = mv;

			if (dis < 0) {
				continue;
			}

			Xt.setVal(i, 0, 0, 0, 0);
			Xt.setVal(j, 1, 0, 1, 0);
			Xt.setVal(dis, 2, 0, 2, 0);
			Xt.setVal(1, 3, 0, 3, 0);

			Matrix recon = Q*Xt;
			double w;
			recon.getData(&w, 3, 0, 3, 0);
			//cout << recon << endl;

			if (w == 0)
				continue;

			recon = recon / w;
			//cout << recon << endl;

			Matrix Xt1 = pose * recon;
			//cout << Xt1 << endl;

			Matrix est = P*Xt1;
			est.getData(&w, 2, 0, 2, 0);
			est = est / w;

			if (w == 0)
				continue;

			double xt1, yt1;
			est.getData(&xt1, 0, 0, 0, 0);
			est.getData(&yt1, 1, 0, 1, 0);

			mv.val[0] = 1;
			mv.val[2] = 64.0 * (xt1 - i) + 32768;
			mv.val[1] = 64.0 * (yt1 - j) + 32768;
			flow.at<Vec3w>(j, i) = mv;
			//cout << est << endl;	


		}
	}

	imwrite("flow.png", flow);
	return 0;
}