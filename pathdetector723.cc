/*
 * pathdetector.cc
 *
 *  Created on: Aug 12, 2017
 *      Author: zhian
 */

#include "pathdetector.h"
#include "d2seamdetector.h"
#include "linemodel.h"
#include "bsplinemodel.h"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <map>
#include <algorithm>
#include "utils.h"
#include <unistd.h>
#include "detectors.h"
#include "bench.h"
#include "linefeature.h"
#include "lineanalysis.h"
using namespace cv;
using namespace std;
using namespace walgo;

namespace walgo {
class ptclose
{
public:
	ptclose(int th) : _th(th) {}
	bool operator() (const Point& a, const Point& b) const
	{
		int dist = std::abs(a.x-b.x);
		if ( dist < _th ) return true;
		return false;
	}
	int _th;
};

class cmp
{
public:
	#include "linemodel.h"
bool operator() (const Point& a, const Point& b) const
	{
		return a.x < b.x;
	}
};

int gsize(const vector<Point>& gr)
{
	cmp c;
	const Point& gmin = *std::min_element(gr.begin(), gr.end(), c);
	const Point& gmax = *std::max_element(gr.begin(), gr.end(), c);
	cout << "gsize: gmin " << gmin << " gmax "<< gmax << endl;
	return gmax.x-gmin.x;
}

Point leftEnd(const vector<Point>& gr)
{
	cmp c;
	return *std::min_element(gr.begin(), gr.end(), c);
}

Point rightEnd(const vector<Point>& gr)
{
	cmp c;
	return *std::max_element(gr.begin(), gr.end(), c);
}

bool cmpHist(const pair<int,int> &x,const pair<int,int> &y)
{
    return x.second > y.second;
}

}

walgo::PathDetector::PathDetector(int lmax,
									walgo::PathDetector::D2PointType d2Type,
									std::string name) :
									_lmax(lmax), _d2PointType(d2Type)
{
	if (!readConfig(name, _config)) {
		cout <<"WARNING! No config.4 file " << endl;
	}
	init();
}

walgo::PathDetector::PathDetector(int lmax,
								walgo::PathDetector::D2PointType d2Type,
								const map<string, int>& config) :
								_lmax(lmax), _d2PointType(d2Type)
{
	_config = config;
	init();
}

void walgo::PathDetector::init()
{
	int xc = 640;
	int yc = 512;
	int w = 1280;
	int h = 1024;
	_cullMethod = 0;
	getConfigEntry(_config, "CULL_METHOD", _cullMethod);
	// must have ROI entries.
	getConfigEntry(_config, "ROI_CENTER_X", xc);
	getConfigEntry(_config, "ROI_CENTER_Y", yc);
	getConfigEntry(_config, "ROI_WIDTH", w);
	getConfigEntry(_config, "ROI_HEIGHT", h);

	//转换成opencv适用的方框格式
	int x = xc - w/2;
	int y = yc -h/2;
	cout << "ROI: "<< x <<" "<< y << "  "<< w << "  "<< h << endl;
	_roi = Rect2d(x,y,w,h);
	_showImage = 0;
	_smin = 0;
	_smax = _lmax;
	getConfigEntry(_config, "SHOW_IMAGE", _showImage);
	_saveImage = 0;
	getConfigEntry(_config, "SAVE_DEBUG_IMAGE", _saveImage);
	_d2mu = Mat::zeros(1280, _lmax, CV_8U);
	_d2mv = Mat::zeros(1024, _lmax, CV_8U);
	_d2mw = Mat::zeros(1000, _lmax, CV_8U);
	_d2mu2nd = Mat::zeros(1280, _lmax, CV_8U);
	_d2mv2nd = Mat::zeros(1024, _lmax, CV_8U);
	_d2mw2nd = Mat::zeros(1000, _lmax, CV_8U);
	_nmax = 0;
	_nmin = _lmax;
	_subMin = 10;
	_subMax = 0;
	_d2vs.resize(_lmax, {});
        cout << "sizeof(_d2vs):" << sizeof(_d2vs) << endl;
	_cogs.resize(_lmax, {});
	_uvec.resize(_lmax, 0);
	_vvec.resize(_lmax, 0);
	_wvec.resize(_lmax, 0);
	_subMap.resize(_lmax, 0);
	_subLMin.resize(10, 1000000);
	_subLMax.resize(10, -1);

	int minSig = 300;
	int maxSig = 330;
	getConfigEntry(_config, "SIGNATURE_MIN", minSig);
	getConfigEntry(_config, "SIGNATURE_MAX", maxSig);
	_minSignal = 0.0001*minSig;
	_maxSignal = 0.0001*maxSig;
        _maxS= 777;
        _minS= 666;
}

bool walgo::PathDetector::addImage(const cv::Mat& image, int l, int isub)
{
	if ( l < _nmin) _nmin = l;
	if ( l > _nmax) _nmax = l;
	if ( l > _subLMax[isub]) _subLMax[isub] = l;
	if ( l < _subLMin[isub]) _subLMin[isub] = l;
	if ( isub > _subMax) _subMax = isub;
	if ( isub < _subMin) _subMin = isub;
	_subMap[l] = isub;
        //cv::Rect2d roi= Rect2d(180,0,988,1024);
        Mat img = image(_roi);
	vector<Vec4i> lines;
	vector<TopoPoint> seam;
	vector<D2> d2v;
	vector<float> cog;
	findD2Seam(img, cog, d2v, lines, seam, _config);
        //hu+
        fstream ftest;
        ftest.open("ftest/"+to_string(l)+"ftest.txt",ios::out);
        int count= 0;
        for(auto cc : cog) {ftest << count << " " << cc << endl; count++;}
        ftest << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ "<<endl;
        for(auto dd : d2v) ftest << dd._i << " " << dd._d2 << endl;
        ftest.close();
        //hu+
	if ( _showImage && l >= _smin && l <= _smax)
	{
		Mat cimage;
		cvtColor(image, cimage, CV_GRAY2BGR);
		for ( int i = 0; i < seam.size(); i++) {
			cout << "Detected Seam at: " << seam[i]._point.x << " " << seam[i]._point.y << endl;
		}
		int xoff = _roi.x;
		int yoff = _roi.y;
		//加上线特征标注用蓝色
		for( size_t i = 0; i < lines.size(); i++ )
		{
			Vec4i l = lines[i];
			line( cimage, Point(l[0]+xoff, l[1]+yoff), Point(l[2]+xoff, l[3]+yoff), Scalar(255,0,0), 1, CV_AA);
		}
		for ( int i = 0; i < seam.size(); i++)
			circle(cimage, seam[i]._point+Point2f(xoff,yoff), 10, Scalar(0,255,0));
		namedWindow("SeamImage", WINDOW_AUTOSIZE);
		//显示蓝色方框，因为颜色格式是BGR,所以第一个数255代表蓝色
		rectangle(cimage, _roi, Scalar(255,0,0), 2,1);
		imshow("SeamImage", cimage);
		waitKey();
	}

	_d2vs[l] = d2v;
	_cogs[l] = cog;

	if ( seam.size() < 2) {
		cout << "findD2Seam did not return two points!" << endl;
		return false;
	}
	if ( _d2PointType == D2POINT_2)
	{
		_d2mu.at<uchar>(round(seam[1]._point.x), l) = 255;
		_d2mv.at<uchar>(round(seam[1]._point.y), l) = 255;
		_uvec[l] = seam[1]._point.x;
		_vvec[l] = seam[1]._point.y;
		_wvec[l] = seam[1]._signal;
	}
	else if ( _d2PointType == D2POINT_1)
	{
		_d2mu.at<uchar>(round(seam[0]._point.x), l) = 255;
		_d2mv.at<uchar>(round(seam[0]._point.y), l) = 255;
		_uvec[l] = seam[0]._point.x;
		_vvec[l] = seam[0]._point.y;
		_wvec[l] = seam[0]._signal;
	}
	else if ( _d2PointType == D2POINT_MID )
	{
		_d2mu.at<uchar>(round((seam[0]._point.x+seam[1]._point.x)*0.5), l) = 255;
		_d2mv.at<uchar>(round((seam[0]._point.y+seam[1]._point.y)*0.5), l) = 255;
		_uvec[l] = (seam[0]._point.x+seam[1]._point.x)*0.5;
		_vvec[l] = (seam[0]._point.y+seam[1]._point.y)*0.5;
		_wvec[l] = seam[0]._signal;
	}
}

bool walgo::PathDetector::detect(cv::Mat& uimg,
                                         vector<float>& uvec,
                                                         AbsLineModel*& model,
                                                         vector<Point>& linePoints,
                                                         vector<vector<Point> >& lineSegments,
                                                         vector<Point2f>& endPoints)
{
	endPoints.clear();
	bench b;
	b.start();
	if ( _cullMethod == 0 )
	{
		LineFeature* linefeature = LineFeature::getLineFeature(HOUGH);
		vector<Vec4i> lines;
		int doEdgeDetCopy = _config["DO_EDGE_DETECTION"];
		// 不做edgedetection
		_config["DO_EDGE_DETECTION"] = 0;
		linefeature->extract(uimg, uimg, lines, _config);
		_config["DO_EDGE_DETECTION"] = doEdgeDetCopy;
		//找到线最集中的角度. 要将线长设够大， 避免将短线计入。
		int angleRange = 0;
		getConfigEntry(_config, "PATH_ANGLE_RANGE", angleRange);
		int binsize = _config["ANGLE_BIN_SIZE"];
                int numbins = (int) floor((360.0/(double)binsize)+0.5);
		vector<double> weights(numbins);
		vector<vector<Vec4i> > hist(numbins);
		calcAngleHistogram(lines, hist, weights, binsize);
		vector<Vec4i>  selectedLines;
		selectMaxAngles(hist, weights, numbins,
				selectedLines, angleRange/binsize);
		//用linemodel找线上的点， 到线模型距离小于 eps
		int lineModelType = 0;
		getConfigEntry(_config, "LINE_MODEL_TYPE", lineModelType);
		if ( lineModelType == 0)
			model = new LineModel(uimg, selectedLines, 2);
		else if ( lineModelType == 1 )
		{
			int nDiv = 4;
			getConfigEntry(_config, "LINE_MODEL_BSPLINE_NDIV", nDiv);
			model = new BsplineModel(uimg, selectedLines, 4, nDiv);
		}
	}
	else
	{
		int imin = 100;
		int imax = 84;
		getConfigEntry(_config, "SIGNATURE_LOWER",imin);
		getConfigEntry(_config, "SIGNATURE_UPPER", imax);
		float div = 0.001*(_maxSignal-_minSignal);
		float minSig = _minSignal + imin*div;
		float maxSig = _minSignal + imax*div;

        if(_minS != 666 && _maxS != 777)
        {
            minSig = _minS;
            maxSig = _maxS;
        }

		cout << "minSig = " << minSig << "  maxSig = " << maxSig << endl;

		for ( int i = _nmin; i < _nmax; i++) {
			//cout << i << "  " << _wvec[i] << endl;
			if ( _wvec[i] > maxSig || _wvec[i] < minSig ) {
				uimg.at<uchar>(round(uvec[i]), i)= 0;
			}
		}

        if (_minS != 666 && _maxS != 777)
        {
            imwrite("filteredU0.5th.png", uimg);
            vector<Point> locations;
            cv::findNonZero(uimg, locations);
            int medianPoint= locations.size()/2;
            int medianU= locations[medianPoint].y;
            vector<Point> badPts;
            for(auto p:locations)
            {
                cout << p.y << " ";
                if(p.y-medianU > 100 || p.y-medianU < -100) badPts.push_back(p);
            }
            for(auto it:badPts)
            {
                remove(locations.begin(),locations.end(),it);
                uimg.at<uchar>(it.y,it.x)= 0;
            }
            _minS= 666;
            _maxS= 777;
        }


		int lineModelType = 0;
		int niter = 4;
		getConfigEntry(_config, "LINE_MODEL_TYPE", lineModelType);
		if ( lineModelType == 0)
			model = new LineModel(uimg, niter);
		else if ( lineModelType == 1 )
		{
			int nDiv = 4;
			getConfigEntry(_config, "LINE_MODEL_BSPLINE_NDIV", nDiv);
			model = new BsplineModel(uimg, nDiv, niter);
		}
	}
	//lm.setUseOrigModel();
	int eps = 3;
	getConfigEntry(_config, "LINE_MODEL_EPS",eps);
	model->build((float) eps);
	//int xmin, xmax, ymin, ymax;
	//lm.getRange(xmin, xmax, ymin, ymax);
	std::vector<cv::Point> pts;
	model->getAllPoints(pts);
	linePoints = pts;   // return points within eps from model line
	//用opencv的partition将线分为线段，距离定义为时间距离，参见ptclose类
	int distTh = 20;   // can be smaller
	getConfigEntry(_config, "PARTITION_DIST", distTh);
	ptclose ptc(distTh);
	vector<int> labels;
        partition(pts, labels, ptc);
	map<int, vector<Point> > groups;
	for ( int i = 0; i < labels.size(); i++ ) {
		groups[labels[i]].push_back(pts[i]);
	}
	vector<vector<Point>> vgr;
	for ( auto && it : groups)
		vgr.push_back(it.second);
	// 把线段按左右排序。
	auto gcmp = [] (const vector<Point>& a, const vector<Point>& b)
	{
		return a[0].x < b[0].x;
	};
	std::sort(vgr.begin(), vgr.end(), gcmp);
	cout << "there are total " << vgr.size() << " groups " << endl;

	if ( vgr.size() == 0) return false;
	//最小长度，和最小点数
	int minEndSection = 30;
	int minEndNumPoints = 5;
	getConfigEntry(_config, "MIN_END_SECTION", minEndSection);
	getConfigEntry(_config, "MIN_END_NUM_POINTS", minEndNumPoints);
	cout << "culling segments from left" << endl;
	auto it = vgr.begin();
	while (it < vgr.end())
	{
		int g = gsize(*it);
		int s = (*it).size();
		cout << "g = " << g << " s = "<< s << endl;
		if (g<minEndSection || s < minEndNumPoints )
			it = vgr.erase(it);
		else
			break;
	}
	cout << "culling segments from right" << endl;
	it = vgr.end();
	while (it > vgr.begin())
	{
		it --;
		int g = gsize(*it);
		int s = (*it).size();
		cout << "g = " << g << " s = "<< s << endl;
		if (g<minEndSection || s < minEndNumPoints )
			it = vgr.erase(it);
		else
			break;
	}
	b.stop();
	lineSegments = vgr;  // return line segments
	cout << "end detection used: " << b.duration() << endl;
	cout << " there are " << vgr.size() << " groups left " << endl;
	if (vgr.size() == 0) return false;
	Point le = leftEnd(vgr[0]);
	cv::Point2f p;
	p.x = le.x;
	p.y = le.y;
	endPoints.push_back(p);
	Point re = rightEnd(vgr[vgr.size()-1]);
	p.x = re.x;
        p.y = re.y;
	endPoints.push_back(p);

	return true;
}

bool walgo::PathDetector::build2ndUVImages(float u0, float v0, float u1, float v1, int lmin, int lmax)
{
        float uavg = (u0+u1)*0.5;
        int range = 150;
        getConfigEntry(_config, "2ND_DETECT_RANGE", range);
        int useAvg = 0;
        getConfigEntry(_config, "2ND_DETECT_USE_AVG", useAvg);
        int umin = uavg-range;
        int umax = uavg+range;
        if ( umin < 0) umin=0;
        if ( umax > 1279) umax = 1279;
        cout << "umin = " << umin << " umax = " << umax << endl;
        if ( lmax == lmin) {
                cout << "Error less than 1 seam point detected!" << std::endl;
                return false;
        }
        float ku = (u1-u0)/(lmax-lmin);
        for ( int l = _nmin; l <= _nmax; l++ )
        {
                if ( !useAvg )
                {
                        uavg = ku*(l-lmin)+u0;
                        umin  = uavg-range;
                        umax = uavg + range;
                        if ( umin < 0) umin=0;
                        if ( umax > 1279) umax = 1279;
                }
		const vector<D2>& d2v = _d2vs.at(l);
		int s = d2v.size();
		if ( s == 0) continue;
		int offset = d2v[0]._i;
		int dumin = umin-offset;
		if ( dumin < 0) dumin = 0;
		int dumax = umax-offset;
		if ( dumax < 0) dumax = 0;
		if ( dumin > (s-1)) dumin = s-1;
		if ( dumax > (s-1)) dumax = s-1;
		if ( dumax == dumin ) continue;
        //cout <<  " offset = " << offset << " dumin = " << dumin << " dumax = " << dumax << endl;
		const D2& d1 = *std::min_element(d2v.begin()+dumin, d2v.begin()+dumax, d2less);
		const D2& d2 = *std::max_element(d2v.begin()+dumin, d2v.begin()+dumax, d2less);
		const vector<float>& cog = _cogs.at(l);
		int u1 = d1._i;
		int u2 = d2._i;
        //cout << " u1 = " << u1 << " u2 = " << u2 << endl;
		int um = (u1+u2+1)/2;
        //cout << " cog[u1] " << cog[u1] << endl;
		int v1 = round(cog[u1]);
		int v2 = round(cog[u2]);
		int vm = (v1+v2+1)/2;
		float sRange = 1.0/(_maxSignal-_minSignal);
		if ( _d2PointType == D2POINT_2)
		{
			_d2mu2nd.at<uchar>(u2,l) = 255;
			_d2mv2nd.at<uchar>(v2,l) = 255;
                        _uvec[l] = u2;
			_vvec[l] = v2;
			_wvec[l] = d2._d2;
			int iw = (d2._d2-_minSignal)*sRange;
			if ( iw < 0) iw = 0; if ( iw > 999 ) iw = 999;
			_d2mw2nd.at<uchar>(iw,l) = 255;
		}
		else if ( _d2PointType == D2POINT_1)
		{
			_d2mu2nd.at<uchar>(u1, l) = 255;
			_d2mv2nd.at<uchar>(v1, l) = 255;
			_uvec[l] = u1;
			_vvec[l] = v1;
			_wvec[l] = d1._d2;
			int iw = (d1._d2-_minSignal)*sRange;
			if ( iw < 0) iw = 0; if ( iw >999 ) iw = 999;
			_d2mw2nd.at<uchar>(iw,l) = 255;
		}
		else if ( _d2PointType == D2POINT_MID )
		{
			_d2mu2nd.at<uchar>(um, l) = 255;
			_d2mv2nd.at<uchar>(vm, l) = 255;
			_uvec[l] = um;
			_vvec[l] = vm;
			_wvec[l] = d1._d2;
			int iw = (d1._d2-_minSignal)*sRange;
			if ( iw < 0) iw = 0; if ( iw >999 ) iw = 999;
			_d2mw2nd.at<uchar>(iw,l) = 255;
		}
	}

}

bool walgo::PathDetector::detect(vector<Point2f>& endPoints,
						 	 	 vector<int>& endSteps,
								 std::vector<cv::Point>& uLinePts,
								 std::vector<cv::Point>& vLinePts,
								 std::vector<std::vector<cv::Point>>& uSegments,
                                                                 std::vector<std::vector<cv::Point>>& vSegments,
                                                                 std::vector<float>& uv_kc)
{
	if ( _showImage )
	{
		string d2mustring = string("D2 U vs T");
		string d2mvstring = string("D2 V vs T");

		namedWindow(d2mustring, WINDOW_NORMAL);
		imshow(d2mustring, _d2mu);
		namedWindow(d2mvstring, WINDOW_NORMAL);
		imshow(d2mvstring, _d2mv);
		waitKey(0);
	}
	int lmin = _nmin;
	int lmax = _nmax;
	cout <<"**** minSignal = " << _minSignal << " maxSignal = " << _maxSignal << endl;

	float sRange = 1000.0/(_maxSignal - _minSignal);
	for ( int nl = _nmin; nl < _nmax; nl++)
	{
		float wval = _wvec[nl];
		int iw = round((wval-_minSignal)*sRange);
		if ( iw < 0) iw =  0;
		if ( iw >= 1000) iw  = 999;
		_d2mw.at<uchar>(iw, nl) = 255;
	}
	endPoints.clear();
	vector<Point2f> upoints, vpoints;
	AbsLineModel* umodel = nullptr;
	AbsLineModel* vmodel = nullptr;

        if ( _saveImage ) {
		imwrite("d2mu.jpg", _d2mu);
		imwrite("d2mv.jpg", _d2mv);
		imwrite("d2mw.jpg", _d2mw);
        }
        int ni= 0;
        getConfigEntry(_config, "NEW_ITER", ni);
        if(ni)
        {
                vector<Point> locations;
                cv::findNonZero(_d2mu, locations);
                fstream fd2topos;
                fd2topos.open("fd2_.txt",ios::out);
                vector<D2> filterd2;
                
                for(auto p : locations)
                {
                    fstream fd2;
                    fd2.open("ftest/d2_"+to_string(p.x)+".txt",ios::out);
                    for(auto d:_d2vs[p.x])
                    {
                        fd2 << d._i << " " << d._d2 << endl;
                        if(d._i == p.y)
                        {
                            fd2topos << p.x << " " << d._i << " " << d._d2 << endl;

                            D2 fd2(d._i,d._d2);
                            filterd2.push_back(fd2);
                        }
                    }
                    fd2.close();
                }
                fd2topos.close();
                cout << _maxSignal << _minSignal << endl;
                int imin = 400;
                int imax = 1000;
                getConfigEntry(_config, "SIGNATURE_LOWER",imin);
                getConfigEntry(_config, "SIGNATURE_UPPER", imax);
                cout << " imin imax: " << imin << " " << imax << endl;
                float div = 0.001*(_maxSignal-_minSignal);
                float minSig = _minSignal + imin*div;
                float maxSig = _minSignal + imax*div;
                float gap= 0.00125;
                int multiple= 3;
                getConfigEntry(_config, "GAP_MULTIPLE", multiple);
                gap*= multiple;
                //cout << minSig << " " << maxSig << " " << gap << endl;
                vector<int> topFive;
                cout << " calc d2 histogram: ";
                int gapNum = (int)((maxSig-minSig)/gap);
                cout << minSig << " " << maxSig << " " << gap << "gapNum= " << gapNum << endl;
                /*
                vector<int> topFiveA;
                topFiveA.resize(gapNum,0);
                for(auto d:filterd2)
                {
                    int cc= (int)((d._d2-minSig)/gap);
                    if(cc > 0 && cc < gapNum) topFiveA[cc]= topFiveA[cc]+1;
                }

                for(int j= 1; j < gapNum; j++) cout << topFiveA[j] << " ";

                vector<int> input;
                for(int i=0; i < gapNum ; i++) input.push_back(topFiveA[i]);
                vector<int> & output= topFive;

                    int len= 5;

                    map<double,int> in;
                    int index = 0;
                    for(auto i :input)
                    {
                        double esp = 0.000001;
                        double d = esp*index +(double)i;
                        in[d] = index;
                        index ++;
                    }
                    map<double,int>::iterator it = in.end();
                    it--;
                    int l = 0;
                    while(it != in.begin() && l<len)
                    {
                        output.push_back(it->second);
                        it--;
                        l++;
                    }

                //cout << " ok ! ";

                sort(topFive.begin(),topFive.end());
                if(topFive[len-1]-topFive[0] < 10)
                {
                    cout << "_minS: " << _minS << " _maxS: " << _maxS << endl;
                    _maxS= minSig+(topFive[len-1]+1)*gap;
                    _minS= minSig+topFive[0]*gap;
                    cout << "_minS: " << _minS << " _maxS: " << _maxS << endl;
                }
                //else cout << "min: " << minSig << " max: " << maxSig << endl;

                for(auto t:topFive) cout << t << " " << topFiveA[t] << " ";
                */
                vector<pair<int,int>> pairVec;
                pair<int,int> pairArray[gapNum];
                vector<int> histVec;
                histVec.resize(gapNum,0);
                for(auto d:filterd2)
                {
                    int cc= (int)((d._d2-minSig)/gap);
                    if(cc > 0 && cc < gapNum) histVec[cc]= histVec[cc]+1;
                }

                for(int j= 1; j < gapNum; j++)
                {
                    cout << histVec[j] << " ";
                    pairArray[j].first= j;
                    pairArray[j].second= histVec[j];
                    pairVec.push_back(pairArray[j]);
                }
                sort(pairVec.begin(),pairVec.end(),cmpHist);
                int len= 5;
                int count= 0;
                for(int i= 0; i < len; i++)
                {
                    cout << pairVec[i].first << " " << pairVec[i].second << endl;
                    topFive.push_back(pairVec[i].first);
                    count+= pairVec[i].second;
                }

                sort(topFive.begin(),topFive.end());
                int ntc= 100;
                getConfigEntry(_config, "NEW_ITER_COUNT", ntc);
                if(topFive[len-1]-topFive[0] < 2*(len+1) && count > ntc)
                {
                    cout << "_minS: " << _minS << " _maxS: " << _maxS << endl;
                    _maxS= minSig+(topFive[len-1]+1)*gap;
                    _minS= minSig+topFive[0]*gap;
                    cout << "new_minS: " << _minS << " new_maxS: " << _maxS << endl;
                }
                else cout << "min: " << minSig << " max: " << maxSig << endl;
	}
	Mat filteredU,filteredV;
	_d2mu.copyTo(filteredU);
    _d2mv.copyTo(filteredV);
	bool usuccess = detect(filteredU, _uvec, umodel, uLinePts, uSegments,  upoints);
	if ( ! usuccess ) {
		if ( umodel) {
			delete umodel;
			umodel = nullptr;
		}
		return false;
	}
	bool vsuccess = detect(filteredV, _vvec, vmodel, vLinePts, vSegments, vpoints);
	// we can tolerate vsuccess false
    if ( _saveImage)
    {
                imwrite("filteredU.png", filteredU);
                imwrite("filteredV.png", filteredV);
    }
	lmin = round(upoints[0].x);
	float u0 = umodel->model(lmin);
	float v0 = vmodel->model(lmin);
	lmax = round(upoints[1].x);
	float u1 = umodel->model(lmax);
	float v1 = vmodel->model(lmax);
	int do2ndDetect = 1;
	getConfigEntry(_config, "DO_2ND_DETECT", do2ndDetect);

	if ( do2ndDetect ){
                build2ndUVImages(u0, v0, u1, v1, lmin, lmax);
		if ( _saveImage ) {
			imwrite("d2mu2nd.jpg", _d2mu2nd);
			imwrite("d2mv2nd.jpg", _d2mv2nd);
			imwrite("d2mw2nd.jpg", _d2mw2nd);
		}
		if (umodel) {delete umodel; umodel = nullptr;}
		if (vmodel) {delete vmodel; vmodel = nullptr;}
		upoints.clear();
		vpoints.clear();
		Mat filteredU, filteredV;
		_d2mu2nd.copyTo(filteredU);
		_d2mv2nd.copyTo(filteredV);
		usuccess = detect(filteredU, _uvec, umodel, uLinePts, uSegments, upoints);
		if ( ! usuccess )   {
			if ( umodel ) {delete umodel; umodel = nullptr; }
			return false;
		}
		vsuccess = detect(filteredV, _vvec, vmodel, vLinePts, vSegments, vpoints);
		if ( _saveImage)
		{
			imwrite("filteredU2nd.jpg", filteredU);
			imwrite("filteredV2nd.jpg", filteredV);
		}
		lmin = round(upoints[0].x);
		u0 = umodel->model(lmin);
		float v0 = vmodel->model(lmin);
		lmax = round(upoints[1].x);
		u1 = umodel->model(lmax);
		float v1 = vmodel->model(lmax);

	}
	int numSubs = _subMax-_subMin+1;
	endSteps.resize(numSubs*2);

	for ( int i = 0; i < numSubs; i++)
	{
		endSteps[i*2] = lmax;
		endSteps[i*2+1] = lmin;
	}

	for ( int i = 0; i< uSegments.size(); i++)
	{
		const vector<Point>& seg = uSegments[i];
		int l1 = leftEnd(seg).x;
		int l2 = rightEnd(seg).x;
		int mysub = -1;
		for ( int j = _subMin; j<=_subMax; j++)
		{
			if ( l1 >=  _subLMin[j] && l2 <= _subLMax[j] )
			{
				mysub = j;
				break;
			}
		}

		if ( mysub == -1 )
		{
			cout << "WARNING: u segment #" << i << " does not belong to any sub seam! " << endl;
			continue;
		}
		else
			cout << " segment " << l1 << " "<< l2 << " belongs to subseam " << mysub << endl;
		mysub = mysub-_subMin;
		if ( l1 < endSteps[mysub*2]) endSteps[mysub*2] = l1;
		if ( l2 > endSteps[mysub*2+1]) endSteps[mysub*2+1] = l2;
	}
	endPoints.clear();

	for ( int i = 0; i < numSubs; i++)
	{
		int l1 = endSteps[i*2];
		int l2 = endSteps[i*2+1];
		float u1 = umodel->model(l1);
		float v1 = vmodel->model(l1);
		float u2 = umodel->model(l2);
		float v2 = vmodel->model(l2);
		endPoints.push_back(Point2f(u1,v1));
		endPoints.push_back(Point2f(u2,v2));
		cout << "subSeam #" << i << " left step " << l1 << " right step " << l2 << endl;
		cout << "Left point " << u1 << "  " << v1 << endl;
		cout << "Right point " <<  u2 << "  " << v2 << endl;
                fstream endspoints;
                endspoints.open("endspoints.txt",ios::out);
                endspoints << l1 << " " << l2 << endl;
                endspoints << u1 << " " << v1 << endl;
                endspoints << u2 << " " << v2 << endl;
                endspoints.close();
        }
        fstream weldPointEnds;
        weldPointEnds.open("weldPointEnds.txt",ios::out);
        for(auto it : uv_kc) weldPointEnds << it << " ";
        weldPointEnds << endl;
        int doDepart = 0;
        getConfigEntry(_config, "DO_DEPART", doDepart);
        if  (doDepart > 0 && !uv_kc.empty())
        {
            int ic= uv_kc.size();
            cout << " ic " << ic << endl;
            int interNum[ic]; //= {225,777,999};
            for(int i; i < ic; i++) {interNum[i]= uv_kc[i]; cout << interNum[i] << " ";}
            uv_kc.clear();
            uv_kc.push_back(umodel->getK());
            uv_kc.push_back(umodel->getC());
            uv_kc.push_back(vmodel->getK());
            uv_kc.push_back(vmodel->getC());
            vector<int> L;
            endSteps.clear();
            for(auto it : uSegments )
            {
                for(auto io : it) L.push_back(io.x);
            }
            for(auto it : L) cout << it << " "; cout << "L.size(): " << L.size() << endl;
            sort(L.begin(),L.end());

            fstream cogLine;
            cogLine.open("cogLine.txt",ios::out);
            //for (int ii= *L.begin(); ii <= *(L.end()-1);ii++)
            for (int ii= 0; ii < _cogs.size();ii++)
            {
                double vlsum= 0;
                double vrsum= 0;
                int Ns= 6;
                int Ne= 26;
                int count= 0;
                if (_cogs[ii].size() > 11)
                {
                    int u= (int)(umodel->model(ii)+0.50001);

                    for(int i= Ns;i < Ne;i++)
                    {   count++;
                        vlsum= _cogs[ii][u-i]+vlsum;
                        vrsum= _cogs[ii][u+i]+vrsum;
                    }
                    cogLine << ii << " " << vlsum/count-vmodel->model(ii) << " " << vrsum/count-vmodel->model(ii) << endl;// "delta_v: " << _cogs[it][umodel->model(it)] - vmodel->model(it) << endl;
                    }
            }
            cogLine.close();
            vector<int> sparpe;
            int eraseSparpe = 0;
            getConfigEntry(_config, "ERASE_SPARPE", eraseSparpe);
            fstream fuv;
            fuv.open("fuvinpd.txt",ios::out);
            for(auto it : L)
            {
                double vlsum= 0;
                double vrsum= 0;
                int Ns= 0;
                int Ne= 36;
                int count= 0;
                int u= (int)(umodel->model(it)+0.50001);

                for(int i= Ns;i < Ne;i++)
                {   count++;
                    vlsum= _cogs[it][u-i]+vlsum;
                    vrsum= _cogs[it][u+i]+vrsum;
                }
                cout << it << " " << vlsum/count-vmodel->model(it) << " " << vrsum/count-vmodel->model(it) << " " << count << endl;// "delta_v: " << _cogs[it][umodel->model(it)] - vmodel->model(it) << endl;
                if (eraseSparpe >0 && (vlsum/count-vmodel->model(it) < eraseSparpe/-35.0 || vrsum/count-vmodel->model(it) < eraseSparpe/-35.0))
                {
                    sparpe.push_back(it);
                    fuv << it << " : " << vlsum/count-vmodel->model(it) << " , " << vrsum/count-vmodel->model(it) << endl;
                }

            }

            if (eraseSparpe > 0)
            {
                for(auto it : sparpe)
                {
                    remove(L.begin(),L.end(),it);
                    L.pop_back();
                }
            }
            for(auto it : L) cout << it << " "; cout << "L.size(): " << L.size() << endl;

            endSteps.push_back(*L.begin());
            for(int i= 0; i < ic; i++)
            {
                endSteps.push_back(*(lower_bound(L.begin(),L.end(),interNum[i])-1));
                endSteps.push_back(*(lower_bound(L.begin(),L.end(),interNum[i]))) ;
            }
            endSteps.push_back(*(L.end()-1));

            for ( int i = 0; i < endSteps.size()/2; i++) weldPointEnds << endSteps[2*i] << " " << endSteps[2*i+1] << endl;
            weldPointEnds.close();
            /*
            system("python findweldpointends.py /home/yongxiang/robotics/image/bin/");

            ifstream ifs("weldPointEnds.txt");
            vector<string> vec_str;
            string str;
            while(getline(ifs,str))
            {
                cout<<str.c_str()<<endl;
                vec_str.push_back(str);
            }
            cout<<"----------------"<<endl;
            cout<<vec_str[vec_str.size()-1]<<endl;
            cout<<"----------------"<<endl;
            str = vec_str[vec_str.size()-1];
            vector<int> vec_dou;
            char* split = " ";
            char* p = strtok(const_cast<char*>(str.c_str()),split);
            while(p!=NULL)
            {
                vec_dou.push_back(atoi(p));
                p = strtok(NULL,split);
            }
            cout<<"--------1--------"<<endl;
            for(auto& data:vec_dou)
            {
                cout<<data<<endl;
            }
            for ( int i = 0; i < endSteps.size(); i++) endSteps[i]= vec_dou[i];
            */
            endPoints.clear();
            for ( int i = 0; i < endSteps.size()/2; i++)
            {
            cout << "part" << i << " start/end: " << endSteps[2*i] << " , " << endSteps[2*i+1] << endl;
            cout << " UV: " << uv_kc[0]*endSteps[2*i]+uv_kc[1] <<  " , " << uv_kc[2]*endSteps[2*i]+uv_kc[3] << " , " << uv_kc[0]*endSteps[2*i+1]+uv_kc[1] << " , " << uv_kc[2]*endSteps[2*i+1]+uv_kc[3] << endl;
            float u1 = umodel->model(endSteps[2*i]);
            float v1 = vmodel->model(endSteps[2*i]);
            float u2 = umodel->model(endSteps[2*i+1]);
            float v2 = vmodel->model(endSteps[2*i+1]);
            endPoints.push_back(Point2f(u1,v1));
            endPoints.push_back(Point2f(u2,v2));
            //fuv << uv_kc[0]*endSteps[2*i]+uv_kc[1] <<  " " << uv_kc[2]*endSteps[2*i]+uv_kc[3] << " " << uv_kc[0]*endSteps[2*i+1]+uv_kc[1] << " " << uv_kc[2]*endSteps[2*i+1]+uv_kc[3]<< " " << endSteps[2*i] << " " << endSteps[2*i+1] << endl;
            }
            //fuv << endSteps << endl;
            fuv << endl << endPoints;
            fuv.close();

        }
        endPoints.clear();
        for ( int i = 0; i < endSteps.size()/2; i++)
        {
        float u1 = umodel->model(endSteps[2*i]);
        float v1 = vmodel->model(endSteps[2*i]);
        float u2 = umodel->model(endSteps[2*i+1]);
        float v2 = vmodel->model(endSteps[2*i+1]);
        endPoints.push_back(Point2f(u1,v1));
        endPoints.push_back(Point2f(u2,v2));
        }

	if ( umodel ) { delete umodel; umodel = nullptr;}
	if ( vmodel ) { delete vmodel; vmodel = nullptr;}

        cout << "get out of pd!" << endl;
	return true;

}

