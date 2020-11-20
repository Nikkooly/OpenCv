#include "lab5.h"
using namespace cv;
using namespace std;
void laba5() {
    cornerPointsHarris();
    cornerPointsTomasi();
    affineTransformations();
}
void cornerPointsHarris() {
    Mat image, gray;
    Mat output, output_norm, output_norm_scaled;
    image = imread("image.jpg", IMREAD_COLOR);
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    // Detecting corners
    output = Mat::zeros(image.size(), CV_32FC1);
    cornerHarris(gray, output, 2, 3, 0.04);
    // Normalizing
    normalize(output, output_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(output_norm, output_norm_scaled);
    // Drawing a circle around corners
    for (int j = 0; j < output_norm.rows; j++) {
        for (int i = 0; i < output_norm.cols; i++) {
            if ((int)output_norm.at<float>(j, i) > 100) {
                circle(image, Point(i, j), 2, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }
    cv::imshow("Harris", image);
    cv::waitKey();
};

void cornerPointsTomasi() {
    Mat image, gray;
    int maxCorners = 500;
    image = imread("image.jpg", 1);
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    maxCorners = MAX(maxCorners, 1);
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    Mat copy = image.clone();
    goodFeaturesToTrack(gray,
        corners,
        maxCorners,
        qualityLevel,
        minDistance,
        Mat(),
        blockSize,
        gradientSize,
        useHarrisDetector,
        k);
    int radius = 4;
    for (size_t i = 0; i < corners.size(); i++)
    {
        circle(copy, corners[i], radius, Scalar(0, 0, 255), FILLED);
    }
    imshow("Shi-Tomasi", copy);
    cv::waitKey();


};

void affineTransformations() {
    Mat src(imread("paperstol.jpg"));
    Point2f srcTri[3];
    srcTri[0] = Point2f(0.f, 0.f);
    srcTri[1] = Point2f(src.cols - 1.f, 0.f);
    srcTri[2] = Point2f(0.f, src.rows - 1.f);
    Point2f dstTri[3];
    dstTri[0] = Point2f(0.f, src.rows * 0.33f);
    dstTri[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
    dstTri[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);
    Mat warp_mat = getAffineTransform(srcTri, dstTri);
    Mat warp_dst = Mat::zeros(src.rows, src.cols, src.type());
    warpAffine(src, warp_dst, warp_mat, warp_dst.size());
    Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
    Point center2 = Point(src.cols / 2, src.rows / 2);
    double angle = -10.0;
    double scale = 1.2;
    Mat rot_mat = getRotationMatrix2D(center, angle, scale);
    Mat rot_mat2 = getRotationMatrix2D(center2, angle, scale);
    Mat warp_rotate_dst;
    Mat rotate_dst;
    warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());
    warpAffine(src, rotate_dst, rot_mat2, src.size());
    imshow("Source image", src);
    imshow("Warp", warp_dst);
    //imshow("Warp + Rotate", warp_rotate_dst);
    imshow("Rotate", rotate_dst);
    waitKey();
};