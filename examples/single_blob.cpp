#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// g++ single_blob.cpp -o single_blob -I/usr/local/include `pkg-config --cflags --libs opencv4` 

using namespace cv;
using namespace std;

int main(int argc, char**argv)
{
    Mat src, hsv, mask;

    src = imread(argv[1], 1);
    if (src.empty())
    {
        cerr << "Error loading image" << endl;
        return -1;
    }

    // Convert to HSV color space
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // Define blue color range in HSV
    // H: 90-110 (blue), S: 100-255 (medium to high saturation), V: 180-255 (high value)
    Scalar lower_blue(90, 140, 180);
    Scalar upper_blue(110, 255, 255);
    inRange(hsv, lower_blue, upper_blue, mask);

    Moments m = moments(mask, true);
    Point p;
    if (m.m00 != 0) {
        p = Point(m.m10/m.m00, m.m01/m.m00);
        circle(src, p, 5, Scalar(0, 0, 0), -1); // Draw centroid on source image
        cout << "Centroid: " << Mat(p) << endl;
    } else {
        cout << "No baby blue region detected." << endl;
    }

    imshow("HSV", hsv);
    imshow("Source", src);
    imshow("Baby Blue Mask", mask);
    waitKey(0);
    return 0;
}