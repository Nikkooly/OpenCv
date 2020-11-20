#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace cv;
using namespace std;


int main(int argc, const char** argv)
{
    VideoCapture capture("video2.mp4");
    VideoWriter outputVideo;
    outputVideo.open("video4.wmv", cv::VideoWriter::fourcc('W', 'M', 'V', '2'), capture.get(CAP_PROP_FPS), Size(640, 480), true);
    Mat frame, image;
    CascadeClassifier detectorBody, detectorFace, detectorEyes;
    double scale = 1;
    //detectorBody.load("case.xml");
    detectorBody.load("haarcascade_fullbody.xml");
    detectorEyes.load("haarcascade_eye_tree_eyeglasses.xml"); 
    detectorFace.load("haarcascade_frontalface_alt.xml");
    for (;;) {
        bool Is = capture.grab();
        if (Is == false)
        {
            cout << "Video Capture Fail" << endl;
            break;
        }
        else
        {
            const clock_t begin_time = clock();
            vector<Rect> humans, faces, eyes;
            Mat img, original;
            capture.retrieve(img, CAP_OPENNI_BGR_IMAGE);
            resize(img, img, Size(640, 480), INTER_LINEAR);
            // Store original colored image
            img.copyTo(original);
            cvtColor(img, img, COLOR_BGR2GRAY);
            equalizeHist(img, img);
            detectorBody.detectMultiScale(img, humans, 1.1, 4);
            for (int gg = 0; gg < humans.size(); gg++) {
                rectangle(original, humans[gg].tl(), humans[gg].br(), Scalar(0, 0, 255), 3);
            
                         detectorFace.detectMultiScale(img, faces);
                         for (size_t i = 0; i < faces.size(); i++)
                         {
                             Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
                             ellipse(original, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
                             Mat faceROI = img(faces[i]);
                             //-- In each face, detect eyes
                             std::vector<Rect> eyes;
                             /* if (faces.size()>0) {
                                  detectorBody.detectMultiScale(img, humans, 1.1, 4);
                                  for (int gg = 0; gg < humans.size(); gg++) {
                                      rectangle(original, humans[gg].tl(), humans[gg].br(), Scalar(0, 0, 255), 3);
                                  }
                              }*/
                             detectorEyes.detectMultiScale(faceROI, eyes);
                             for (size_t j = 0; j < eyes.size(); j++)
                             {
                                 Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                                 int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                                 circle(original, eye_center, radius, Scalar(255, 0, 0), 4);
                             }

                             clock_t diff = clock() - begin_time;
                             // convert time into string
                             char buffer[126];
                             sprintf_s(buffer, "%d", diff);
                             // display TIME ms on original image
                             putText(original, buffer, Point(100, 20), 1, 2, Scalar(255, 255, 255), 2, 8, 0);
                             putText(original, "ms", Point(150, 20), 1, 2, Scalar(255, 255, 255), 2, 8, 0);
                             // draw results
                             namedWindow("prew", WINDOW_AUTOSIZE);
                             imshow("prew", original);
                             outputVideo << original;
                             int key1 = waitKey(20);
                         }
               }                  
            
        }

    }
    return 0;
}
