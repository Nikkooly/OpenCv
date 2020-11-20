//#define CV_RETR_EXTERNAL 0 // найти только крайние внешние контуры
//#define CV_RETR_LIST     1 // найти все контуры и разместить их списком
//#define CV_RETR_CCOMP    2 // найти все контуры и разместить их в виде 2-уровневой иерархии
//#define CV_RETR_TREE     3 // найти все контуры и разместить их в иерархии вложенных контуров

//#define CV_CHAIN_CODE               0 // цепной код Фридмана
//#define CV_CHAIN_APPROX_NONE        1 // все точки цепного кода переводятся в точки
//#define CV_CHAIN_APPROX_SIMPLE      2 // сжимает горизонтальные, вертикальные и диагональные сегменты и оставляет только их конечные точки
//#define CV_CHAIN_APPROX_TC89_L1     3 // применяется алгоритм
//#define CV_CHAIN_APPROX_TC89_KCOS   4 // аппроксимации Teh-Chin
//#define CV_LINK_RUNS                5 // алгоритм только для CV_RETR_LIST

//метод аппроксимации режим поиска
#include "lab4.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <minmax.h>
#include <opencv2\core\core_c.h>
#include <opencv2\imgproc\imgproc_c.h>
using namespace cv;
using namespace std;
void laba4() {
    contours();
    lines();
    circles();
}

void contours() {
    const int N = 2;
    cv::String names[]{ "Initial Image", "Image Contour" };
    Mat img0 = imread("image2.jpg", COLOR_RGB2GRAY);
    Mat img1;
    cvtColor(img0, img1, 11);
    Canny(img1, img1, 100, 200);
    vector<vector<Point>> contours;
    findContours(img1, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
    // CV_FILLED fills the connected components found
    drawContours(mask, contours, -1, Scalar(255), FILLED);
    Mat crop(img0.rows, img0.cols, CV_8UC3);
    crop.setTo(Scalar(255, 255, 255));
    img0.copyTo(crop, mask);
    normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);
    //imshow("mask", mask);
    waitKey();
    cv::Mat images[]{ img0, crop };
    end(N, names, images);
}

void lines() {
    const int N = 3;
    cv::String names[]{ "Initial Image", "Detected Lines (in red) - Standard Hough Line Transform","Detected Lines (in red) - Probabilistic Line Transform" };
    Mat dst, cdst, cdstP;
    Mat src = imread("second_image.jpg", IMREAD_GRAYSCALE);
    Canny(src, dst, 50, 200, 3);
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    Mat images[]{ src, cdst, cdstP };
    end(N, names, images);
}

void circles() {
    Mat image, gray;
    image = imread("image13.jpg", IMREAD_COLOR);
    cvtColor(image, gray, COLOR_BGR2GRAY);
    // Smoothing the image to get rid of noise, so no false detections
    // Here we used a 11x11 filter
    blur(gray, gray, Size(7, 7), Point(-1, -1)); 
    // Store the detected circles in a 3d-vector
    vector<Vec3f> circles;
    for (int maxR = 0; maxR < 200; maxR = maxR + 10)
    {
        // Apply hough transform
        HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
           gray.rows/16, // detect circle with different distance
            70, 90,
            20, maxR);
        // Draw the detected circles
        for (size_t i = 0; i < circles.size(); i++)
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // Draw the center point of the circle
            circle(image, center, 3, Scalar(0, 255, 0), -1, 8, 0);
            // Draw the circle shape
            circle(image, center, radius, Scalar(0, 255, 0), 3, 8, 0);
        }
    }
    cv::imshow("circles", image);
    cv::waitKey(0);

//gray: Input image(grayscale).
//circles : A vector that stores sets of 3 values : xc, yc, r for each detected circle.
//HOUGH_GRADIENT : Define the detection method.Currently this is the only one available in OpenCV.
//dp = 1 : The inverse ratio of resolution.
//min_dist = gray.rows / 16 : Minimum distance between detected centers.
//param_1 = 200 : Upper threshold for the internal Canny edge detector.
//param_2 = 100 * : Threshold for center detection.
//min_radius = 0 : Minimum radius to be detected.If unknown, put zero as default.
//max_radius = 0 : Maximum radius to be detected.If unknown, put zero as default.
    //
}

