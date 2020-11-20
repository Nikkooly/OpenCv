#include "util.h"

void openWindows(int n, cv::String* names, cv::Mat* images) {
    for (int i(0); i < n; ++i) {
        cv::namedWindow(*(names + i));
        imshow(*(names + i), *(images + i));
    }
}

void closeAndReleaseWindows(int n, cv::String* names, cv::Mat* images) {
    for (int i(0); i < n; ++i) {
        cv::destroyWindow(*(names + i));
        (images + i)->release();
    }
}

void end(int n, cv::String* names, cv::Mat* images) {
    openWindows(n, names, images);
    cv::waitKey(0);
    closeAndReleaseWindows(n, names, images);
}