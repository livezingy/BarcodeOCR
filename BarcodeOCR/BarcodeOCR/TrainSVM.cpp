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


#include "TrainSVM.h"
#include <conio.h>
#include "util.h"
#include "lbp.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;
const string pathLabels1 = "label1_";
const string pathLabels2 = "label2_";
const string path_NoLabels = "noLabel_";
extern string svmPath;
extern int numLabel1;
extern int numLabel2;
extern int numNoLabels;
extern Ptr<cv::ml::SVM> svmClassifier;
extern string svmFilename;
extern string descriptorType;

TrainSVM::TrainSVM() 
{
	imageWidth = 400;
	imageHeight = 200;
}

TrainSVM::~TrainSVM() 
{  
}


void TrainSVM::trainClassifierAuto(const Ptr<cv::ml::TrainData>& trainData)
{
	// SVM learning algorithm
	clock_t begin_time = clock();

	svmClassifier->setType(cv::ml::SVM::C_SVC);
	svmClassifier->setKernel(cv::ml::SVM::KernelTypes::RBF);
	svmClassifier->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 40000, 0.00001));
	svmClassifier->trainAuto(trainData, 10, cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C),
		cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P),
		cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU), cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF),
		cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE), true);

	fprintf(stdout, ">> Saving model file...\n");

	svmClassifier->save(svmFilename);

	fprintf(stdout, ">> Your SVM Model was saved to %s\n", svmFilename);

	float timer = ( clock () - begin_time ) /  CLOCKS_PER_SEC;
	cout << "Time: " << timer << endl;
}

void TrainSVM::generateLabelDataset(VideoCapture cap, int numData, int nClass)
{
	DetectLabel detectLabels;
	detectLabels.showBasicImages = true;
	vector<Mat> label;
	Mat normalImage;
	string path_data;
	int i = 0;


	if (nClass == 1)
	{
	    path_data = pathLabels1;
		numLabel1 = numData;
	}
	else if (nClass == 2)
	{
		path_data = pathLabels2;
		numLabel2 = numData;
	}
	else
	{
		path_data = path_NoLabels;
		numNoLabels = numData;
	}

	while (i < numData)
	{
		cap >> normalImage;
		detectLabels.segment(normalImage, label);
		// TODO: RGB to gray
		if (label.size() > 0)
		{
			if (!label[0].empty())
			{
				stringstream ss;
				ss << path_data << i << ".jpg";
				imwrite(ss.str(), label[0]);
				cout << "path = "<< ss.str() << endl;
				i++;
			}
		}
		namedWindow("normalImage",WINDOW_NORMAL);
		imshow("normalImage",normalImage);
		label.clear();
		if(waitKey(30) >= 0) break;
	}
}

void TrainSVM::labelToXml()
{
    Mat classes;//(numLabel1+numLabel2+numNoLabels, 1, CV_32FC1);
    Mat trainingData;//(numLabel+numNoLabels, imageWidth*imageHeight, CV_32FC1 );

    Mat trainingImages;
    vector<int> trainingLabels;

    cout << numLabel1 << endl;
    cout << path_NoLabels << endl;

    for(int i=0; i< numLabel1; i++)
    {
        stringstream ss(stringstream::in | stringstream::out);
        ss << pathLabels1 << i << ".jpg";
        cout << "read path = "<< ss.str() << endl;
        Mat img=imread(ss.str(), 0);
        img= img.reshape(1, 1);
        trainingImages.push_back(img);
        trainingLabels.push_back(1);
    }

    for(int i=0; i< numLabel2; i++)
    {
        stringstream ss(stringstream::in | stringstream::out);
        ss << pathLabels2 << i << ".jpg";
        cout << "read path 2= "<< ss.str() << endl;
        Mat img=imread(ss.str(), 0);
        img= img.reshape(1, 1);
        trainingImages.push_back(img);
        trainingLabels.push_back(2);
    }

    for(int i=0; i< numNoLabels; i++)
    {
        stringstream ss(stringstream::in | stringstream::out);
        ss << path_NoLabels << i << ".jpg";
        cout << "read path 3= "<< ss.str() << endl;
        Mat img=imread(ss.str(), 0);
        if (img.empty()) break;
        img= img.reshape(1, 1);
        trainingImages.push_back(img);
        trainingLabels.push_back(0);

    }

    Mat(trainingImages).copyTo(trainingData);
    //trainingData = trainingData.reshape(1,trainingData.rows);
    trainingData.convertTo(trainingData, CV_32FC1);
    Mat(trainingLabels).copyTo(classes);



	auto train_data = cv::ml::TrainData::create(trainingData, cv::ml::SampleTypes::ROW_SAMPLE,
		Mat(classes));

	trainClassifierAuto(train_data);

	/*
    FileStorage fs(svmPath, FileStorage::WRITE);
    fs << "TrainingData" << trainingData;
    fs << "classes" << classes;
    fs.release();
	*/
}


void TrainSVM::getTrainSetFromCamera(void)
{
    VideoCapture cap(1); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
    {
		cout << "Error! camera open failed!" << endl;
	}
	else
	{
		bool isExit = false;
		while (!isExit)
		{
			Utils::print_file_lines("etc/TrainSet_menu");
			std::cout << "Please make a choice:";

			char select = -1;
			bool isRepeat = true;
			while (isRepeat)
			{
				std::cin >> select;
				switch (select)
				{
				case '1':
					generateLabelDataset(cap, numLabel1, 1);
					break;
				case '2':
					generateLabelDataset(cap, numLabel2, 2);
					break;
				case '3':
					generateLabelDataset(cap, numNoLabels, 0);
					break;
				case '4':
					labelToXml();
					break;
				case '5':
					isRepeat = false;
					isExit = true;
					break;

				default:
					std::cout << "input error, Please re-input:";
					break;
				}
				if ((waitKey(30) >= 0))
				{
					break;
				}
			}
		}


	}
}

void TrainSVM::getTrainSetFromLocal(void)
{
	//std::vector<int> responses;
	Mat classes;//(numLabel1+numLabel2+numNoLabels, 1, CV_32FC1);
	cv::Mat samples;
	vector<string> imgPathTrain;

	vector<string> tmpImgTrain;

	char buffer[260] = { 0 };

	sprintf(buffer, "IMAGES/label1/train");
	imgPathTrain = utils::getFiles(buffer);
	vector<int> objPresentTrain(imgPathTrain.size(), 1);

	sprintf(buffer, "IMAGES/label2/train");
	tmpImgTrain = utils::getFiles(buffer);

	for (auto file : tmpImgTrain)
	{
		imgPathTrain.push_back(file);
		objPresentTrain.push_back(2);
	}

	sprintf(buffer, "IMAGES/noLabel/train");
	tmpImgTrain = utils::getFiles(buffer);

	for (auto file : tmpImgTrain)
	{
		imgPathTrain.push_back(file);
		objPresentTrain.push_back(0);
	}

	for (auto f : imgPathTrain) {
		auto image = cv::imread(f,0);
		if (!image.data) {
			fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.c_str());
			continue;
		}
		

		if (descriptorType == "LBP")
		{
			cv::Mat feature;
			getLBPFeatures(image, feature);
			samples.push_back(feature);
		}
		else
		{
			image = image.reshape(1, 1);
			samples.push_back(image);
		}		
	}
	samples.convertTo(samples, CV_32FC1);
	//Mat(objPresentTrain).copyTo(classes);

	auto train_data = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE,
		Mat(objPresentTrain));

	trainClassifierAuto(train_data);
}

void TrainSVM::getLBPFeatures(const cv::Mat& image, cv::Mat& features)
{

	//Mat grayImage;
//	cvtColor(image, grayImage, CV_RGB2GRAY);
	
	Mat lbpimage;
	lbpimage = libfacerec::olbp(image);

	//spatial_histogram函数返回的特征至为单行且数据类型为CV_32FC1
	Mat lbp_hist = libfacerec::spatial_histogram(lbpimage, 32, 4, 4);

	features = lbp_hist;
	
}
