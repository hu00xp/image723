/*
 * tracker.cc
 *
 *  Created on: Nov 1, 2016
 *      Author: zhian
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "bench.h"
#include "utils.h"
#include "pathdetector.h"

using namespace cv;
using namespace std;
using namespace walgo;

template<typename T>
int count(T& x);

int main(int argc, char** argv)
{
	if (argc < 4  || argc == 5)
	{
		cout << "usage: " << argv[0] << " <directory> lmin lmax" << endl;
		cout << "or " << argv[0] << " <directory> lmin lmax smin smax" << endl;
		cout << "where smin smax is range of images you want to display" << endl;
		exit(0);
	}

    Mat frame;
    string fname;
    std::map<std::string, int> config;
    readConfig(std::string("config.4"), config);
    string directory = argv[1];
    int nmin = atoi(argv[2]);
    cv::Rect2d roi= Rect2d(0,0,1280,1024);

    vector<Vec4i> lines;
    vector<TopoPoint> seam;
    vector<D2> d2v;
    vector<float> cog;
/*
    bench b;
    fname = string(directory)+string("/")+to_string(nmin)+string(".jpg");
    //读取下个图形
    cout << "Reading " << fname << "..... "<< endl;
    frame = imread(fname, 0);

    if ( !(frame.data))
    {
        cout << "can't read file " << fname << endl;
        exit(1);
    }
    b.start();
    Mat img = frame(roi);
    findD2Seam(img, cog, d2v, lines, seam, config);
    b.stop();
    cout << "addImage used: " << b.duration() << endl;
*/

	int nmax = atoi(argv[3]);
	int* lsubs=new int[4];
	int lsubssize;
	PathDetector::D2PointType t;
    if ( argc == 8)
    {
    lsubssize= 4;
    t= PathDetector::D2POINT_2;
     }
    else if ( argc == 7)
    {
    lsubssize= 3;
    t= PathDetector::D2POINT_2;
    }
    else if (argc == 5)
    {
        lsubssize= 1;
        t= PathDetector::D2POINT_2;
    }
    for(int i= 0; i < lsubssize; i++) lsubs[i] =atoi(argv[4+i]);
   
  // int lsubs [] = {210, 333,229, 220};
	
	PathDetector pd(nmax, t);//PathDetector::D2POINT_2);
	    
    int smin = nmin;
    int smax = nmax;
	if ( argc == 6)
	{
		smin = atoi(argv[4]);
		smax = atoi(argv[5]);
		pd.setShowImage();
		pd.setShowImageIndices(smin, smax);
	}
    
	int l = 0;
	string seamName;
	for (int nsub = 0; nsub < lsubssize; nsub++)
	{
                if ( argc == 7 || argc == 5) 		seamName = string("seam3_part")+to_string(nsub);
    else  seamName = string("seam5_part")+to_string(nsub);
	for (int nl=0; nl < lsubs[nsub]; nl++)
	{
		bench b;
		fname = string(directory)+string("/")+seamName+string("/")+to_string(nl)+string(".jpg");
		//读取下个图形
		cout << "Reading " << fname << "..... "<< endl;
		frame = imread(fname, 0);

		if ( !(frame.data))
		{
	      cout << "can't read file " << fname << endl;
			exit(1);
	    }
		b.start();
		pd.addImage(frame, l, nsub);
		++l;
		b.stop();
		cout << "addImage used: " << b.duration() << endl;
	}
	}
	vector<Point2f> epts;
	vector<Point> uLinePts, vLinePts;
	vector<vector<Point>> uSegments, vSegments;
	vector<int> endSteps;
        vector<float> uv_kc;
        pd.detect(epts, endSteps, uLinePts, vLinePts, uSegments, vSegments,uv_kc);
        if (!uv_kc.empty())
        {
        for(auto it : uv_kc )
            cout << it << " ";
        cout << "##DO_DEPART$$ " <<  endl;
        }
        cout << " Seam start/end: ";
	for ( int i = 0; i < endSteps.size()/2; i++)
                cout << endSteps[i*2] << "  " << endSteps[i*2+1] << "  ";
	cout << endl;



//delete []lsubs;
return 0;
}

template<typename T>
int count(T& x)
{
    int s1 = sizeof(x);
    int s2 = sizeof(x[0]);
    int result = s1 / s2;
    return result;
}
