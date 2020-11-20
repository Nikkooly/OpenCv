#include "lab3.h"
#include <iostream>
#include <conio.h>
using namespace std;
using namespace cv;
void laba3() {
    char ch;
    int code;
    cv::Mat frame;
    cv::VideoCapture cap;
    int deviceID = 0;
    int apiID = cv::CAP_ANY;
    cap.open(deviceID + apiID);

   while (cap.read(frame)) {
       //ch = getch();
       sobel(&frame);
        //laplas(&frame);
        //canny(&frame);

        imshow("Live", frame);
        if (cv::waitKey(27) >= 0)
            break;
        /*code = static_cast<int>(ch);
        if (ch == 27) // если клавиша esc
            break;*/
    }
}

void sobel(cv::Mat* mat) {
    int ddepth = CV_16S;
    double alpha = 0.5, beta = 0.5;
    cv::Mat grayImg, xGrad, yGrad, xGradAbs, yGradAbs, grad;

    GaussianBlur(*mat, *mat, Size(3, 3), 0, 0, BORDER_DEFAULT);     // преобразование в оттенки серого     
    cvtColor(*mat, grayImg, 11);     // вычисление производных по двум направлениям     
    Sobel(grayImg, xGrad, ddepth, 1, 0); // по Ox     
    Sobel(grayImg, yGrad, ddepth, 0, 1); // по Oy     // преобразование градиентов в 8-битные     
    convertScaleAbs(xGrad, xGradAbs);
    convertScaleAbs(yGrad, yGradAbs);     // поэлементное вычисление взвешенной      // суммы двух массивов     
    addWeighted(xGradAbs, alpha, yGradAbs, beta, 0, *mat);
}



void laplas(cv::Mat* mat) {
    cv::Mat grayImg, laplacianImg, laplacianImgAbs;
    int ddepth = CV_16S;

    //GaussianBlur(*mat, *mat, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cvtColor(*mat, grayImg, 11);
    Laplacian(grayImg, laplacianImg, ddepth);
    convertScaleAbs(laplacianImg, *mat);
}

void canny(cv::Mat* mat) {
    cv::Mat grayImg;
    double lowThreshold = 10, uppThreshold = 100;

    blur(*mat, *mat, cv::Size(3, 3));
    cvtColor(*mat, grayImg, 11);
    Canny(grayImg, *mat, lowThreshold, uppThreshold);
}