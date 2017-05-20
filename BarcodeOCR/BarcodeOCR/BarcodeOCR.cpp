/* --------------------------------------------------------
* author：livezingy
*
* BLOG：http://www.livezingy.com
*
* Development Environment：
*      Visual Studio V2013
*      opencv3.1
*      Tesseract3.04
*
* Version：
*      V1.0    20170220
--------------------------------------------------------- */

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ml.h>
#include <opencv2\opencv.hpp>

#include <fstream>
#include <iostream>
#include <math.h>
#include <string.h>
#include <time.h>
#include "stdafx.h"
#include <conio.h>

#include "util.h"
#include "DetectLabel.h"
#include "LabelOCR.h"
#include "TrainSVM.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

Ptr<cv::ml::SVM> svmClassifier;
const string paramsFile = "params.xml";
string svmPath;
string labelPath;
string descriptorType;
string svmFilename;
int numLabel1 = 100;
int numLabel2 = 100;
int numNoLabels = 100;
int noLabel;

DetectLabel detectLabels;
LabelOCR labelOcr;
TrainSVM trainSVM;

void getTestImageFromVideo(void);
void getTestImageFromLocal(void);
void chooseTestImage(void);
void chooseTrainImage(void);
void getSVM(void);

int main(int argc, char** argv)
{
	bool isExit = false;
	while (!isExit) 
	{
		Utils::print_file_lines("etc/svm_menu");
		std::cout << "Please make a choice:";

		char select = -1;
		bool isRepeat = true;
		while (isRepeat) 
		{
			std::cin >> select;
			isRepeat = false;
			switch (select) {
			case '1':
				descriptorType = "ALL";
				svmFilename = "svm/" + descriptorType + ".xml.gz";
				break;
			case '2':
				descriptorType = "LBP";
				svmFilename = "svm/" + descriptorType + ".xml.gz";
				break;
			case '3':
				isExit = true;
				break;

			default:
				std::cout << "input error, Please re-input:";
				isRepeat = true;
				break;
			}
		}
		if (isExit == false)
		{
			getSVM();
		}
		
	}
	return 0;
}

void getSVM(void)
{
	FileStorage fs(svmFilename, FileStorage::READ);

	if (fs.isOpened())
	{
		cout << "*** LOADING SVM CLASSIFIER ***" << endl;
		svmClassifier = cv::ml::StatModel::load<cv::ml::SVM>(svmFilename);

		fs.release();
	}
	else
	{
		chooseTrainImage();
	}
	
	if (svmClassifier == NULL)
	{
		std::cout << "get SVM file failed.";
	}
	else
	{
		chooseTestImage();
	}
}

void chooseTrainImage(void)
{
	bool isExit = false;
	while (!isExit)
	{
		Utils::print_file_lines("etc/source_menu");
		std::cout << "Please choose the Train source image:";

		char select = -1;
		bool isRepeat = true;
		while (isRepeat) 
		{
			std::cin >> select;
			isRepeat = false;
			switch (select)
			{
			case '1':
				trainSVM.getTrainSetFromCamera();
				isExit = true;
				break;
			case '2':
				trainSVM.getTrainSetFromLocal();
				isExit = true;
				break;
			case '3':
				isExit = true;
				break;

			default:
				std::cout << "input error, Please re-input:";
				isRepeat = true;
				break;
			}
		}
	}

}

/*
The test Images could get from camera or local files according user's choice
*/
void chooseTestImage(void)
{
	bool isExit = false;
	while (!isExit)
	{
		Utils::print_file_lines("etc/source_menu");
		std::cout << "Please choose the Test source image:";

		char select = -1;
		bool isRepeat = true;
		while (isRepeat)
		{
			std::cin >> select;
			isRepeat = false;
			switch (select)
			{
			case '1':
				getTestImageFromVideo();
				break;
			case '2':
				getTestImageFromLocal();
				break;
			case '3':
				isExit = true;
				break;

			default:
				std::cout << "input error, Please re-input:";
				isRepeat = true;
				break;
			}
		}
	}
}



void getTestImageFromLocal(void)
{
	//SVM要做的事情只是判断当前的标签所属类别:类别1/类别2/无标签
	//因此在准备测试数据时,与EasyPR类似,分别将类别1,类别2,无标签的数据放置在不同的文件夹中,
	//取数据时即可得到当前图像的类别.
	Mat classes;//(numLabel1+numLabel2+numNoLabels, 1, CV_32FC1);
	cv::Mat samples;
	vector<string> imgPathTrain;

	vector<string> tmpImgTrain;

	char buffer[260] = { 0 };

	sprintf(buffer, "IMAGES/label1/test");
	imgPathTrain = utils::getFiles(buffer);
	vector<char> objPresentTest(imgPathTrain.size(), 1);

	sprintf(buffer, "IMAGES/label2/test");
	tmpImgTrain = utils::getFiles(buffer);

	for (auto file : tmpImgTrain)
	{
		imgPathTrain.push_back(file);
		objPresentTest.push_back(2);
	}

	sprintf(buffer, "IMAGES/noLabel/test");
	tmpImgTrain = utils::getFiles(buffer);

	for (auto file : tmpImgTrain)
	{
		imgPathTrain.push_back(file);
		objPresentTest.push_back(0);
	}

	double count_all = imgPathTrain.size();
	double ptrue_rtrue = 0;
	double ptrue_rfalse = 0;

	vector<Mat> label_1, label_2;
	vector<string> labelText1, labelText2;
	label_1.clear();
	label_2.clear();

	cv::Mat feature;
	int imgIndex = 0;
	for (size_t imageIdx = 0; imageIdx < count_all; imageIdx++)
	{
		auto image = cv::imread(imgPathTrain[imageIdx], 0);

		if (!image.data)
		{
			fprintf(stdout, ">> Invalid image: %s  ignore.\n", imgPathTrain[imageIdx].c_str());
			continue;
		}
		
		if (descriptorType == "LBP")
		{
			trainSVM.getLBPFeatures(image, feature);
		}
		else
		{
			feature = image.reshape(1, 1);
			feature.convertTo(feature, CV_32FC1); // CV_32FC1
		}
		int predict = (int)svmClassifier->predict(feature);

		auto real = objPresentTest[imageIdx];

		if (predict == real)
		{
			ptrue_rtrue++;

			if (1 == real)
			{
				label_1.push_back(image);
			}
			else if (2 == real)
			{
				label_2.push_back(image);
			}
		}
		if (predict != real)
		{
			ptrue_rfalse++;
		}


		cout << "Class: " << predict << endl;
	}

	std::cout << "count_all: " << count_all << std::endl;
	std::cout << "ptrue_rtrue: " << ptrue_rtrue << std::endl;
	std::cout << "ptrue_rfalse: " << ptrue_rfalse << std::endl;



	double precise = 0;
	if (ptrue_rtrue + ptrue_rfalse != 0)
	{
		precise = ptrue_rtrue / (ptrue_rtrue + ptrue_rfalse);
		std::cout << "precise: " << precise << std::endl;
	}
	else
	{
		std::cout << "precise: "
			<< "NA" << std::endl;
	}

	namedWindow("Source Image");
	namedWindow("Binary Image");
	namedWindow("Recognise Image");

	if (label_1.size() > 0)
	{
		labelText1 = labelOcr.runRecognition(label_1, 1);
	}

	if (label_2.size() > 0)
	{
		labelText2 = labelOcr.runRecognition(label_2, 2);
	}

	cin >> buffer;
}

void getTestImageFromVideo(void)
{
	VideoCapture cap(0); //open the default camera
	if (!cap.isOpened())
	{
		cout << "Error! camera open failed!" << endl;
	}
	else
	{
		Mat normalImage, modImage, cropImage1, labelImage1;
		Mat cropImage2, labelImage2, binImage;
		vector<Point> contour;
		vector<vector<Point> > contours;
		Rect label1ROI;

		string text1, text2;

		vector<Mat> possible_labels, label_1, label_2;
		vector<string> labelText1, labelText2;
		detectLabels.showBasicImages = true;
		detectLabels.showAllImages = true;

		namedWindow("normal", WINDOW_NORMAL);

		while (true)
		{
			string optionKey;
			cin >> optionKey;
			if (optionKey == "ESC")
			{
				break;
			}
			cap >> normalImage; // get a new frame from camera
			imshow("normal", normalImage);

			possible_labels.clear();
			label_1.clear();
			label_2.clear();

			// segmentation
			detectLabels.segment(normalImage, possible_labels);

			int posLabels = possible_labels.size();

			if (posLabels > 0)
			{
				//For each possible label, classify with svm if it's a label or no
				for (int i = 0; i< posLabels; i++)
				{
					if (!possible_labels[i].empty())
					{
						Mat gray;
						cvtColor(possible_labels[i], gray, COLOR_RGB2GRAY);
						Mat p = gray.reshape(1, 1);
						p.convertTo(p, CV_32FC1); // CV_32FC1
						int response = (int)svmClassifier->predict(p);
						cout << "Class: " << response << endl;
						if (response == 1)
							label_1.push_back(possible_labels[i]);
						if (response == 2)
							label_2.push_back(possible_labels[i]);
					}
				}
			}

			if (label_1.size() > 0)
			{
				labelText1 = labelOcr.runRecognition(label_1, 1);
			}

			if (label_2.size() > 0)
			{
				labelText2 = labelOcr.runRecognition(label_2, 2);
			}


			if (waitKey(30) >= 0) break;
		}
	}
}
