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

#include "LabelOCR.h"
#include <iostream>

using namespace cv;
LabelOCR::LabelOCR() 
{
    // constructor
    // Pass it to Tesseract API
	tess = new tesseract::TessBaseAPI();
    tess->Init(NULL, "eng", tesseract::OEM_DEFAULT);
	tess->SetVariable("tessedit_char_whitelist", "ABCDEFHIJKLMNOPQRSTUVWXYZ0123456789-"); //-
	tess->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);


    showImages = true;
}

LabelOCR::~LabelOCR() {
    //destructor

    tess->Clear();
	tess->End();
}

void LabelOCR::filterUndesiredChars(std::string &str){
    char chars[] = "?";

    for (unsigned int i = 0; i < strlen(chars); ++i)
    {
        // you need include <algorithm> to use general algorithms like std::remove()
        str.erase (std::remove(str.begin(), str.end(), chars[i]), str.end());
    }
}

void LabelOCR::preProcess(const cv::Mat &InputImage, cv::Mat &binImage)
{
	cv::Mat midImage, midImage2, dst;
	
	//可以获取常用的结构元素的形状:矩形MORPH_RECT(包括线形)、椭圆 MORPH_ELLIPSE(包括圆形)及十字形 MORPH_CROSS
	cv::Mat Morph = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(1, 1));
	
	cv::Mat HPKernel = (cv::Mat_<float>(5, 5) << -1.0, -1.0, -1.0, -1.0, -1.0,
                                        -1.0,  -1.0, -1.0, -1.0,  -1.0,
                                        -1.0,  -1.0, 25.0, -1.0,  -1.0,
                                        -1.0,  -1.0, -1.0, -1.0,  -1.0,
                                        -1.0,  -1.0, -1.0, -1.0,  -1.0);
										
	//中值滤波将图像的每个像素用邻域 (以当前像素为中心的正方形区域)像素的 中值 代替 。
    medianBlur(InputImage, dst, 3);
	
	//
	cv::filter2D(dst, binImage,InputImage.depth(), HPKernel);
	//cv::cvtColor(midImage2, binImage, cv::COLOR_RGB2GRAY);
    //threshold(midImage, binImage, 60, 255, CV_THRESH_BINARY);
    //threshold(binImage, binImage ,0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    //erode(binImage, binImage, 3, Point(-1, -1), 2, 1, 1);
    //morphologyEx( binImage,binImage,MORPH_CLOSE, Morph);
}

std::string LabelOCR::runPrediction1(const cv::Mat &labelImage, int i){

	std::string t1;
    if (labelImage.empty())
        return (t1);

	cv::Mat textImage;
	cv::Mat drawImage = labelImage.clone();

    double labelROI_x = labelImage.cols*0.10; // initial point x
    double labelROI_y = labelImage.rows*0.76; // initial point y
    double labelROI_w = labelImage.cols*0.6;  // width
    double labelROI_h = labelImage.rows*0.20; // heigth

	cv::Rect labelROI(labelROI_x, labelROI_y, labelROI_w, labelROI_h);

	cv::Mat midImage;
    preProcess(drawImage, textImage);


	tess->TesseractRect(textImage.data, 1, textImage.step1(), labelROI.x, labelROI.y, labelROI.width, labelROI.height);
    // Get the text
	char* text1 = tess->GetUTF8Text();
	t1 = std::string(text1);
    if (t1.size() > 2)
        t1.resize(t1.size() - 2);

	std::cout << "label_" << i << ": " << t1 << std::endl;
		
    if (showImages){
		putText(drawImage, t1, cv::Point(labelROI.x + 7, labelROI.y - 5), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 255), 2, 8); // CV_FONT_HERSHEY_SIMPLEX
		rectangle(drawImage, labelROI, cv::Scalar(0, 0, 255), 2, 8, 0);
        //
		imshow("Source Image", labelImage);
		imshow("Binary Image", textImage);
		imshow("Recognise Image", drawImage);
		waitKey(3000);
    }

    return (t1);
}

std::string LabelOCR::runPrediction2(const cv::Mat &labelImage, int i){

	std::string t1;
    if (labelImage.empty())
        return (t1);

	cv::Mat textImage;
	cv::Mat drawImage = labelImage.clone();

    double labelROI_x = labelImage.cols*0.15; // initial point x
    double labelROI_y = labelImage.rows*0.20; // initial point y
    double labelROI_w = labelImage.cols*0.5;  // width
    double labelROI_h = labelImage.rows*0.15; // heigth

	cv::Rect labelROI(labelROI_x, labelROI_y, labelROI_w, labelROI_h);

	cv::Mat midImage;
    preProcess(drawImage, textImage);


	tess->TesseractRect(textImage.data, 1, textImage.step1(), labelROI.x, labelROI.y, labelROI.width, labelROI.height);
    // Get the text
	char* text1 = tess->GetUTF8Text();
	t1 = std::string(text1);
    if (t1.size() > 2)
        t1.resize(t1.size() - 2);

	std::cout << "label_" << i << ": " << t1 << std::endl;

    if (showImages)
	{
		imshow("label_", labelImage);
		
		putText(drawImage, t1, cv::Point(labelROI.x + 7, labelROI.y - 5), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 255), 2, 8); // CV_FONT_HERSHEY_SIMPLEX
		rectangle(drawImage, labelROI, cv::Scalar(0, 0, 255), 2, 8, 0);
        //
		//std::stringstream ss; ss << i;
		//std::string str = ss.str();

		imshow("Source Image", labelImage);
		imshow("Binary Image", textImage);
		imshow("Recognise Image", drawImage);
		waitKey(3000);
	//	namedWindow("Source Image");
	//	namedWindow("Binary Image");
	//	namedWindow("Recognise Image");
    }

    return (t1);
}

std::vector<std::string> LabelOCR::runRecognition(const std::vector<cv::Mat> &labelImage, int labelType)
{
	std::vector<std::string> output;

    output.resize(labelImage.size());

    for( size_t i = 0; i < labelImage.size(); i++ )
    {
        if ( !labelImage[i].empty() && labelType == 1)
            output[i] = runPrediction1(labelImage[i],i);
        if ( !labelImage[i].empty() && labelType == 2)
            output[i] = runPrediction2(labelImage[i],i);
    }
    return (output);
}
