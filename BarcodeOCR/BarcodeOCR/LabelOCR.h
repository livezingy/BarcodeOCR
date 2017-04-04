/*
 * LabelOCR.h
 *
 *  Created on: May 1, 2014
 *      Author: chd
 */

#ifndef LABELOCR_H_
#define LABELOCR_H_

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

#include <iostream>
#include <math.h>
#include <string.h>
#include <sstream>

#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <leptonica/allheaders.h>


class LabelOCR {

public:
    LabelOCR();
    virtual ~LabelOCR();
	std::vector<std::string> LabelOCR::runRecognition(const std::vector<cv::Mat> &labelImage, int labelType);
    tesseract::TessBaseAPI *tess;
    bool showImages;

private:
	void preProcess(const cv::Mat &InputImage, cv::Mat &binImage);
	std::string runPrediction1(const cv::Mat &labelImage, int i);
	std::string runPrediction2(const cv::Mat &labelImage, int i);
	void skeletonize(cv::Mat& im);
	void thinningIteration(cv::Mat& im, int iter);
	void filterUndesiredChars(std::string &str);

};

#endif /* LABELOCR_H_ */
