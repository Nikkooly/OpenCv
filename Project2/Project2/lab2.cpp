#include "lab2.h"
#include "lab3.h"
using namespace cv;
void laba2(cv::String imagePath) {
    convolution1(imagePath);
    blur2(imagePath);
    erosionAndDilatation3(imagePath);
    canny4(imagePath);
    gisto(imagePath);
}

void convolution1(cv::String imagePath) {
    const int N = 2;
    cv::String names[]{ "Initial Image", "Filter2D" };
    // константы для хранения ядра фильтр
    const float kernelData[]{ -1.0f, -1.0f, -1.0f, -1.0f, 9.0f, -1.0f, -1.0f, -1.0f, -1.0f };
    const cv::Mat kernel(3, 3, CV_32FC1, (float*)kernelData);
    cv::Mat imageRGB = cv::imread(imagePath, 1), filter2dImage;

    filter2D(imageRGB, filter2dImage, -1, kernel);

    cv::Mat images[]{ imageRGB, filter2dImage };

    end(N, names, images);
}

void blur2(cv::String imagePath) {
    const int N = 5;
    cv::String names[]{ "Initial Image", "Blur2D", "boxFilter", "gaussianBlur", "medianBlur" };
    cv::Mat imageRGB(cv::imread(imagePath, 1)), blurImage, boxFilterImage, gaussianBlurImage, medianBlurImage;

    cv::blur(imageRGB, blurImage, cv::Size(3, 3));
    boxFilter(imageRGB, boxFilterImage, -1, cv::Size(3, 3));
    GaussianBlur(imageRGB, gaussianBlurImage, cv::Size(3, 3), 1.0);
    medianBlur(imageRGB, medianBlurImage, 1);

    cv::Mat images[]{ imageRGB, blurImage, boxFilterImage, gaussianBlurImage, medianBlurImage };

    end(N, names, images);
}

void erosionAndDilatation3(cv::String imagePath) {
    const int N = 3;
    cv::String names[]{ "Initial Image", "erode", "dilate" };
    cv::Mat imageRGB(cv::imread(imagePath, 1)), erodeImg, dilateImg, element = cv::Mat();

    erode(imageRGB, erodeImg, element);
    dilate(imageRGB, dilateImg, element);

    cv::Mat images[]{ imageRGB, erodeImg, dilateImg, element };

    end(N, names, images);
}

void canny4(cv::String imagePath) {
    const int N = 2;
    cv::String names[]{ "Initial Image", "Canny" };
    cv::Mat img(cv::imread(imagePath, 1)), cannyImg(img);
    double lowThreshold = 70, uppThreshold = 260;

    canny(&cannyImg);

    cv::Mat images[]{ img, cannyImg };

    end(N, names, images);
}
void gisto(cv::String imagePath) {
    const char* initialWinName = "Initial Image", * equalizedWinName = "Flattening histogram";
    Mat grayImg, equalizedImg;
    Mat img(imread(imagePath, 1));
    cvtColor(img, grayImg, 11); 
    equalizeHist(grayImg, equalizedImg);
    namedWindow(initialWinName, WINDOW_AUTOSIZE);   
    namedWindow(equalizedWinName, WINDOW_AUTOSIZE); 
    imshow(initialWinName, grayImg);   
    imshow(equalizedWinName, equalizedImg);
    waitKey();
}