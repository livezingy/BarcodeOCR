/*
 * DetectLabel.h
 *
 *  Created on: May 1, 2014
 *      Author: chd
 */

#ifndef DETECTLABEL_H_
#define DETECTLABEL_H_

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>

#include <iostream>
#include <math.h>
#include <string.h>
#include <sstream>

//using namespace cv;
//using namespace std;

const double PI = 3.14159265359;

struct LabelRegion {
	cv::Mat labelImage;
	cv::Mat cropImage;
    std::string text;
};

class DetectLabel {

public:
    DetectLabel();
    virtual ~DetectLabel();

	void binariza(const cv::Mat &InputImage, cv::Mat &binImage);
	void findRect(const cv::Mat &binImage, std::vector<std::vector<cv::Point> > &mark);
	void createLabelMat(const cv::Mat &normalImage, std::vector<cv::Point> &contour, cv::Mat &labelImage);
	void cropLabelImage(const cv::Mat &normalImage, std::vector<cv::Point> &contour, cv::Mat &cropImage);
	bool verifySize(std::vector<cv::Point> &contour);
    void runDetection();
	void segment(const cv::Mat &InputImage, std::vector<cv::Mat> &output);
    //
    bool showBasicImages;
    bool showAllImages;

private:
	double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0);
	cv::Point getCenter(std::vector<cv::Point> points);
	float distanceBetweenPoints(cv::Point p1, cv::Point p2);
	std::vector<cv::Point> sortCorners(std::vector<cv::Point> square);
	void cropImageWithMask(const cv::Mat &img_orig, const cv::Mat &mask, cv::Mat &crop);
	void cropImageColor(const cv::Mat &img, const cv::Mat &cropImage, cv::Mat & color_crop);
	cv::Scalar regionAvgColor(const cv::Mat &img, const cv::Mat &mask);
	bool regionIsCloseToWhite(const cv::Mat &img, const cv::Mat &mask);
	std::vector<cv::Point> setReducedSquareContour(std::vector<cv::Point> points);

    //
	cv::Mat blankImage;
	std::vector<std::vector<cv::Point> > segments;
    int MaxNumLabels;
    int labelCounter;

};

#endif /* DETECTLABEL_H_ */
