/*
 * detectors.cc
 *
 *  Created on: Apr 10, 2016
 *      Author: zhian
 */
#include <immintrin.h>
#include "detectors.h"
#include "utils.h"
using namespace cv;
using namespace std;
using namespace walgo;
vector<detector*> detector::_obj(NUM_DETECTORS);

detector* detector::getDetector(LASER_DETECTOR_TYPE t)
{
	if ( _obj[t]== NULL)
	{
		if ( t == THRESH )
			_obj[t] = new threshDetector();
		else if ( t == PEAK )
			_obj[t] = new peakDetector();
		else if ( t == LINE )
			_obj[t] = new lineDetector();
		else if ( t == PEAK1D )
			_obj[t] = new Peak1dDetector();
	}
	return _obj[t];
}

//simple threshold
bool threshDetector::detect(const Mat& input, Mat& output, std::map<std::string, int>& params)
{
	int th = 20;
	std::map<std::string, int>::iterator it;
	it = params.find("THRESH_THRESH");
	if (it == params.end())
	{
		cout << "Can't find parameter for THRESH_THRESH for threshold detector" << endl;
	}
	else
	{
		th = (*it).second;
	}
	//cout <<"Threshold is: " << th << endl;
	threshold(input, output, th, 255, 0);
	return true;
}

// peak detector
bool peakDetector::detect(const Mat& input, Mat& output, std::map<std::string, int>& params)
{
	cout << "input type is: " <<  input.type() << endl;
	int thresh = 20;
	std::map<std::string, int>::iterator it;
	it = params.find("PEAK_THRESH");
	if (it == params.end())
	{
		cout << "Can't find parameter for PEAK_THRESH for threshold detector" << endl;
	}
	else
	{
		thresh = (*it).second;
	}
	cout <<"Threshold is: " << thresh << endl;
	output = input.clone();
	thresh = thresh*9;
	uchar max = 0;
	int useLocalMax = 0;
	useLocalMax = params["PEAK_LOCALMAX"];
	cout << "Peak detector: useLocalMax = " << useLocalMax << endl;
	for ( int i = 1; i < input.rows-1; i++)
	{
		for (int j = 1; j < input.cols-1; j++)
		{

			uchar y  = input.at<uchar>(i,j);
			if ( useLocalMax )
			{
				max = std::max(y, input.at<uchar>(i,j-1));
				max = std::max(max, input.at<uchar>(i+1,j-1));
				max = std::max(max, input.at<uchar>(i-1,j-1));
				max = std::max(max, input.at<uchar>(i+1,j));
				max = std::max(max, input.at<uchar>(i-1,j));
				max = std::max(max, input.at<uchar>(i,j+1));
				max = std::max(max, input.at<uchar>(i-1,j+1));
				max = std::max(max, input.at<uchar>(i+1,j+1));
				if ( y != max )
				{
					output.at<uchar>(i,j) = 0;
					continue;
				}
			}
			short int x = (short) input.at<uchar>(i-1,j-1) +
					(short) input.at<uchar>(i,j-1) +
					(short) input.at<uchar>(i+1,j-1) +
					(short) input.at<uchar>(i-1,j) +
					(short) input.at<uchar>(i,j) +
					(short) input.at<uchar>(i+1,j) +
					(short) input.at<uchar>(i-1,j+1) +
					(short) input.at<uchar>(i,j+1) +
					(short) input.at<uchar>(i+1,j+1);
			if ( y*9-x > thresh)
			{
				//cout << "found peak" << endl;
				output.at<uchar>(i,j) = 255;
			}
			else
				output.at<uchar>(i,j) = 0;
		}
		//cout << endl;
	}
	return true;
}

static float f1v[3][3] =
			{
			{-1, 2, -1},
			{ -1, 2, -1},
			{ -1, 2, -1}
			};

static float f1h[3][3] =
			{
			{-1, -1, -1},
			{2, 2, 2},
			{-1, -1, -1}
			};

static float f1p[3][3] =
	       {
		   {-1, -1, 2},
		   {-1, 2, -1},
		   {2, -1, -1}
	       };
static float f1n[3][3] =
			{
			{2, -1, -1},
			{-1, 2, -1},
			{-1, -1, 2}
			};

static float f3v[5][5] =
			{
			{-2, 1, 2, 1, -2},
			{-2, 1, 2, 1, -2},
			{-2, 1, 2, 1, -2},
			{-2, 1, 2, 1, -2},
			{-2, 1, 2, 1, -2}
			};

static float f3h[5][5] =
			{
			{-2, -2, -2, -2, -2},
			{ 1,  1,  1,  1,  1},
			{ 2,  2,  2,  2,  2},
			{ 1,  1,  1,  1,  1},
			{-2, -2, -2, -2, -2},
			};

static float f3p[5][5] =
			{
			{-2, -2, -1,  1,  2},
			{-2, -1,  1,  2,  1},
			{-1,  1,  2,  1, -1},
			{ 1,  2,  1, -1, -2},
			{ 2,  1, -1, -2, -2},
								};
static float f3n[5][5] =
			{
			{ 2,  1, -1, -2, -2},
			{ 1,  2,  1, -1, -2},
			{-1,  1,  2,  1, -1},
			{-2, -1,  1,  2,  1},
			{-2, -2, -1,  1,  2},
			};


bool lineDetector::generateKernel(Mat& kernel, int angle, int thickness)
{
	int kernelSize = thickness + 2;
	if ( thickness == 1)
	{
		if ( angle == 0 ) kernel = Mat(3,3, CV_32F, f1h);
		else if ( angle == 90 ) kernel = Mat(3,3,CV_32F, f1v);
		else if ( angle == 45 ) kernel = Mat(3,3,CV_32F, f1p);
		else if ( angle == 135) kernel = Mat(3,3,CV_32F, f1n);
		else {
			cout << "Unsupported angle for kernel generation" << endl;
			return false;
		}
	}
	else if ( thickness == 3)
	{
		if ( angle == 0 ) kernel = Mat(5,5, CV_32F, f3h);
		else if ( angle == 90 ) kernel = Mat(5,5,CV_32F, f3v);
		else if ( angle == 45 ) kernel = Mat(5,5,CV_32F, f3p);
		else if ( angle == 135) kernel = Mat(5,5,CV_32F, f3n);
		else {
			cout << "Unsupported angle for kernel generation" << endl;
			return false;
		}
	}
	else {
		cout << "unsupported line thickness " << endl;
		return false;
	}
	return true;
}

//
//using matched filter for lines.
//
bool lineDetector::detect(const Mat& input, Mat& output, map<string, int>& params)
{
	int angle = params["LINE_ANGLE"];
	cout << "line detector angle is: " << angle << endl;
	int thickness = params["LINE_THICKNESS"];
	cout << "line detector thickness: " << thickness << endl;
	Mat kernel, tmpImage;
	if ( !(generateKernel(kernel, angle, thickness)))
	{
		cout << "Unable to generate kernel for angle and thickness " << endl;
		return false;
	}
	Point anchor = Point(thickness, thickness);
	float delta = 0.0;
	int depth = -1;
	filter2D(input, tmpImage, depth, kernel, anchor, delta, BORDER_DEFAULT);
	float th;
	std::map<std::string, int>::iterator it;
	it = params.find("LINE_THRESH");
	if (it == params.end())
	{
		cout << "Can't find parameter for LINE_THRESH for threshold detector" << endl;
	}
	else
	{
		th = (*it).second;
	}
	cout <<"Threshold is: " << th << endl;
	threshold(tmpImage, output, th, 255, 0);

	return true;
}

bool Peak1dDetector::detect(const Mat& input, Mat& output, map<string, int>& config)
{
	//if(__builtin_cpu_supports("avx2")) {
        if ( 1 ) {
                //cout << "假设 cpu 支持 avx2！" << endl;
                detectAVX(input, output, config);
	} else {
		int th = 15;
		getConfigEntry(config, "PEAK1D_THRESH", th);
		output=input.clone();
		output.setTo(0);
		vector<int> maxv;
		maxv.resize(input.cols, 0);

		for ( int i = 0; i < input.rows; i++)
		{
			for (int j = 0; j < input.cols; j++)
			{
				if (input.at<uchar>(i,j) > maxv[j])
					maxv[j] = input.at<uchar>(i,j);
			}
		}
		for ( int i = 0; i < input.rows; i++)
		{
			for (int j = 0; j < input.cols; j++)
			{
				if (input.at<uchar>(i,j) >= maxv[j] && maxv[j] >= th)
					output.at<uchar>(i,j) = 255;
			}
		}
	}
	return true;
}




