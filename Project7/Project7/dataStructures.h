#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>

struct LidarPoint { //single lidar point in space
	double x, y, z, r;
};

struct BoundingBox { // bounding box around a classifies object (contains both 2D and 3D data)
	int boxID; //unique identifier for this bounding box
	int trackID; //unique identifier for the track to which this bounding box belongs

	cv::Rect roi; //2D region of interests in image coordinates
	int classID; //ID based on class file provider to YOLO framework
	double confidence; //classification trust

	std::vector<LidarPoint> lidarPoints; //lidar 3D points which project into 2D image roi
	std::vector<cv::KeyPoint> keypoints; //keypoints encloses by  2D roi
};
#endif /* dataStructures_h */