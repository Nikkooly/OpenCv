#pragma once
#include <opencv2/opencv.hpp>

void openWindows(int n, cv::String* names, cv::Mat* images);

void closeAndReleaseWindows(int n, cv::String* names, cv::Mat* images);

void end(int n, cv::String* names, cv::Mat* images);
