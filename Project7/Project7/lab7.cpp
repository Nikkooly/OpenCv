#include "lab7.h"
#include <iostream>
#include <conio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/optflow/motempl.hpp>
#include "opencv2/imgcodecs.hpp"
#include <fstream>
#include <numeric>
#include "dataStructures.h"


using namespace std;
using namespace cv;
using namespace cv::motempl;
using namespace cv::dnn;

////motion History parametrs
// various tracking parameters (in seconds)
const double MHI_DURATION = 5;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)

// ring image buffer
vector<Mat> buf;
int last = 0;

// temporary images 
Mat mhi, orient, mask, segmask, zplane;
vector<Rect> regions;


void track(Ptr<Tracker> tracker, string trackerType);
void  update_mhi(const Mat& img, Mat& dst, int diff_threshold);


void laba7() {
    int choose;
    for (;;) {
        cout << "\nChoose function:\n 1-Motion History\n 2-Lukas-Canade\n 3-Tracking\n 4-Yolo\n";
        cin >> choose;
        switch (choose) {
        case 1:
            motionHistory();
            break;
        case 2:
            lukasCanade();
            break;
        case 3:
            tracking();
            break;
        case 4:
            yolo();
            break;
        default:
            cout << "Error. Are you stupid?";
            break;
        }
    }
}
void motionHistory() {
    VideoCapture cap("video.mp4");
    buf.resize(2);
    Mat image, motion;
    for (;;)
    {
        cap >> image;
        if (image.empty())
            break;

        update_mhi(image, motion, 30);
        //imshow("Image", image);
        imshow("Motion", motion);

        if (waitKey(10) >= 0)
            break;
    }
}

void lukasCanade() {
    VideoCapture capture("video.mp4");
    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }
    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;
    // Take first frame and find corners in it made by N.Yarmolik
    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    // Create a mask image for drawing purposes made by N.Yarmolik
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    while (true) {
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        // calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);
        vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                // draw the tracks
                line(mask, p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }
        Mat img;
        add(frame, mask, img);
        imshow("Frame", img);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        // Now update the previous frame and previous points made by N.Yarmolik
        old_gray = frame_gray.clone();
        p0 = good_new;
    }
}

void tracking() {
    string trackerType;

    Ptr<Tracker> tracker;
    int chooseType;
    cout << "\nChoose type:\n 1-BOOSTING \n 2-MIL \n 3-KCF \n 4-TLD \n 5-MEDIANFLOW \n 6-GOTURN \n 7-MOSSE \n 8-CSRT \n";
    cin >> chooseType;
        switch (chooseType) {
        case 1:
            tracker = TrackerBoosting::create();
            track(tracker, "BOOSTING");
            break;
        case 2:
            tracker = TrackerMIL::create();
            track(tracker, "MIL");
            break;
        case 3:
            tracker = TrackerKCF::create();
            track(tracker, "KCF");
            break;
        case 4:
            tracker = TrackerTLD::create();
            track(tracker, "TLD");
            break;
        case 5:
            tracker = TrackerMedianFlow::create();
            track(tracker, "MEDIANFLOW");
            break;
        case 6:
            tracker = TrackerGOTURN::create();
            track(tracker, "GOTURN");
            break;
        case 7:
            tracker = TrackerMOSSE::create();
            track(tracker, "MOSSE");
            break;
        case 8:
            tracker = TrackerCSRT::create();
            track(tracker, "CSRT");
            break;
        default:
            cout << "Error. Check input information";
            break;
        }
}

void yolo() {
    //VideoCapture capture("video.mp4");
    Mat img = imread("image.jpg");
    string yoloClassesFile = "coco.names";
    string yoloModelConfiguration = "yolov3.cfg";
    string yoloModelWeights = "yolov3.weights";

    vector<string> classes;
    ifstream ifs(yoloClassesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    //load neural network made by N.Yarmolik
    Net net = readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    //generate 4D blob from input image

    Mat blob;
    double scalefactor = 1 / 255.0;
    Size size = Size(416, 416);
    Scalar mean = Scalar(0, 0, 0);
    bool swapRB = false;
    bool crop = false;
    blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);

    //Get names of output layers

    vector<String> names;
    vector<int> outLayers = net.getUnconnectedOutLayers();
    vector<String> layersNames = net.getLayerNames();

    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) { //get the names of the output layers in names made by N.Yarmolik
        names[i] = layersNames[outLayers[i] - 1];
    }

    //invoke forward propagation through network
    vector<Mat> netOutput;
    net.setInput(blob);
    net.forward(netOutput, names);

    //Scan through all bounding boxes and keep only the ones with high confidence
    float confThreshold = 0.40;
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    for (size_t i = 0; i < netOutput.size(); ++i) {
        float* data = (float*)netOutput[i].data;
        for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols) {
            Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
            Point classId;
            double confidence;

            //Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classId);
            if (confidence > confThreshold) {
                Rect box;
                int cx, cy;
                cx = (int)(data[0] * img.cols);
                cy = (int)(data[1] * img.rows);
                box.width = (int)(data[2] * img.cols);
                box.height = (int)(data[3] * img.rows);
                box.x = cx - box.width / 2;//left
                box.y = cy - box.height / 2;//top

                boxes.push_back(box);
                classIds.push_back(classId.x);
                confidences.push_back((float)confidence);

            }
        }
    }
    //perform non-maxima suppression
    float nmsThreshold = 0.2; //non maximum suppression threshold
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    std::vector<BoundingBox> bBoxes;
    for (auto it = indices.begin(); it != indices.end(); ++it) {
        BoundingBox bBox;
        bBox.roi = boxes[*it];
        bBox.classID = classIds[*it];
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)bBoxes.size(); //zero-based unique identifier for this bounding box

        bBoxes.push_back(bBox);
    }


    //show results
    Mat visImg = img.clone();
    for (auto it = bBoxes.begin(); it != bBoxes.end(); ++it) {
        //Draw rectangle displaying the bounfind box
        int top, left, width, height;
        top = (*it).roi.y;
        left = (*it).roi.x;
        width = (*it).roi.width;
        height = (*it).roi.height;
        rectangle(visImg, Point(left, top), Point(left + width, top + height), Scalar(0, 255, 0), 2);

        string label = format("%.2f", (*it).confidence);
        label = classes[((*it).classID)] + ":" + label;

        //Display label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_ITALIC, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        //rectangle(visImg, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width)), Scalar(0, 255, 255), 2);
        putText(visImg, label, Point(left, top), FONT_ITALIC, 0.75, Scalar(0, 0, 0), 1);
    }
    string windowName = "Object classification";
    namedWindow(windowName, 0.5);
    imshow(windowName, visImg);
    waitKey(0);
}
//motion History
void  update_mhi(const Mat& img, Mat& dst, int diff_threshold)
{
    double timestamp = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds made by N.Yarmolik
    Size size = img.size();
    int i, idx1 = last;
    Rect comp_rect;
    double count;
    double angle;
    Point center;
    double magnitude;
    Scalar color;

    // allocate images at the beginning or
    // reallocate them if the frame size is changed
    if (mhi.size() != size)
    {
        mhi = Mat::zeros(size, CV_32F);
        zplane = Mat::zeros(size, CV_8U);

        buf[0] = Mat::zeros(size, CV_8U);
        buf[1] = Mat::zeros(size, CV_8U);
    }

    cvtColor(img, buf[last], COLOR_BGR2GRAY); // convert frame to grayscale made by N.Yarmolik

    int idx2 = (last + 1) % 2; // index of (last - (N-1))th frame
    last = idx2;

    Mat silh = buf[idx2];
    absdiff(buf[idx1], buf[idx2], silh); // get difference between frames

    threshold(silh, silh, diff_threshold, 1, THRESH_BINARY); // and threshold it
    cv::motempl::updateMotionHistory(silh, mhi, timestamp, MHI_DURATION); // update MHI

    // convert MHI to blue 8u image made by N.Yarmolik
    mhi.convertTo(mask, CV_8U, 255. / MHI_DURATION, (MHI_DURATION - timestamp) * 255. / MHI_DURATION);

    Mat planes[] = { mask, zplane, zplane };
    merge(planes, 3, dst);

    // calculate motion gradient orientation and valid orientation mask
    calcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

    // segment motion: get sequence of motion components
    // segmask is marked motion components map. It is not used further
    regions.clear();
    segmentMotion(mhi, segmask, regions, timestamp, MAX_TIME_DELTA);

    // iterate through the motion components,
    // One more iteration (i == -1) corresponds to the whole image (global motion)
    for (i = -1; i < (int)regions.size(); i++) {

        if (i < 0) { // case of the whole image made by N.Yarmolik
            comp_rect = Rect(0, 0, size.width, size.height);
            color = Scalar(255, 255, 255);
            magnitude = 100;
        }
        else { // i-th motion component
            comp_rect = regions[i];
            if (comp_rect.width + comp_rect.height < 100) // reject very small components
                continue;
            color = Scalar(0, 0, 255);
            magnitude = 30;
        }

        // select component ROI
        Mat silh_roi = silh(comp_rect);
        Mat mhi_roi = mhi(comp_rect);
        Mat orient_roi = orient(comp_rect);
        Mat mask_roi = mask(comp_rect);

        // calculate orientation
        angle = calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION);
        angle = 360.0 - angle;  // adjust for images with top-left origin

        count = norm(silh_roi, NORM_L1);; // calculate number of points within silhouette ROI

        // check for the case of little motion made by N.Yarmolik
        if (count < comp_rect.width * comp_rect.height * 0.05)
            continue;

        // draw a clock with arrow indicating the direction
        center = Point((comp_rect.x + comp_rect.width / 2),
            (comp_rect.y + comp_rect.height / 2));

        circle(img, center, cvRound(magnitude * 1.2), color, 3, 16, 0);
        line(img, center, Point(cvRound(center.x + magnitude * cos(angle * CV_PI / 180)),
            cvRound(center.y - magnitude * sin(angle * CV_PI / 180))), color, 3, 16, 0);
    }
}
//Tracking
void track(Ptr<Tracker> tracker, string trackerType) {
    // Read video
    VideoCapture video("video.mp4");

    Mat frame;
    bool ok = video.read(frame);

    // Define initial bounding box 
    Rect2d bbox(287, 23, 86, 320);
    // Display bounding box. 
    rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

    imshow("Tracking", frame);
    tracker->init(frame, bbox);

    while (video.read(frame))
    {
        // Update the tracking result made by N.Yarmolik
        bool ok = tracker->update(frame, bbox);

        if (ok)
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        else
            putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        putText(frame, trackerType + " Tracker", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

        // Display frame.
        imshow("Tracking", frame);
        // Exit if ESC pressed.
        int k = waitKey(1);
        if (k == 27)
        {
            break;
        }
    }
}


