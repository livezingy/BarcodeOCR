#ifndef TRAINSVM_H_
#define TRAINSVM_H_

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>
#include <sstream>
#include "DetectLabel.h"
//#include "util.h"
//#include "lbp.hpp"
#include "stdafx.h"
#include <opencv2\opencv.hpp>

//using namespace std;
//using namespace cv;
//using namespace cv::ml;

class TrainSVM 
{
public:
    TrainSVM();
    virtual ~TrainSVM();
	
	//int numLabel1=100;
	//int numLabel2=100;
	//int numNoLabels=200;
	int imageWidth;
	int imageHeight;
	
	void trainClassifierAuto(const cv::Ptr<cv::ml::TrainData>& trainData);
	void getTrainSetFromCamera(void);
	void getTrainSetFromLocal(void);
	void getLBPFeatures(const cv::Mat& image, cv::Mat& features);
private:
    
	void generateLabelDataset(cv::VideoCapture cap, int numData, int nClass);
	void labelToXml();
	
};

#endif
/* TRAINSVM_H_ */