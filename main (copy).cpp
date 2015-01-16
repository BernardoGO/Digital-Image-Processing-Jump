#include "opencv2/core/core.hpp"
#include <opencv/highgui.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv/cv.hpp"
#include <iostream>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include <opencv/highgui.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv/cv.hpp"
#include <iostream>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

#define MAX_COUNT 500
char rawWindow[] = "Raw Video";
char opticalFlowWindow[] = "Optical Flow Window";
char imageFileName[32];
long imageIndex = 0;
char keyPressed;

int main() {
const string& filename = "833878_v2.avi";
 VideoCapture cap(filename);
CvCapture *capture = cvCaptureFromAVI("833878_v2.avi");

if(!cap.isOpened()){
    std::cout<<"cannot read video!\n";
    return -1;
}
  namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

  int iLowH = 5;
 int iHighH = 56;//152

  int iLowS = 2;
 int iHighS = 60;

  int iLowV = 15;
 int iHighV = 255;

int xis = 4;
int yip = 5;
int ze = 6;
int aga = 8;

  //Create trackbars in "Control" window

 cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
 cvCreateTrackbar("HighH", "Control", &iHighH, 179);

  cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
 cvCreateTrackbar("HighS", "Control", &iHighS, 255);

  cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
 cvCreateTrackbar("HighV", "Control", &iHighV, 255);

cvCreateTrackbar("xis", "Control", &xis, 255); //Value (0 - 255)
 cvCreateTrackbar("yip", "Control", &yip, 255);
cvCreateTrackbar("ze", "Control", &ze, 255); //Value (0 - 255)
 cvCreateTrackbar("aga", "Control", &aga, 255);

int framenum = 0;
bool bSuccess = false;


    while (true)
    {
    IplImage* frame = NULL;
 Mat imgOriginal;
frame = cvQueryFrame(capture);//capture.read(imgOriginal);
imgOriginal = frame;
    framenum++;
        // read a new frame from video

if (!frame) //if not success, break loop
        {
             cout << "Cannot read a frame from video stream" << endl;
             cvSetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO , 0);
             continue;
        }

    Mat imgHSV;

   cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
imgHSV = imgHSV(Rect(0,400,1280,200));
//GaussianBlur( imgHSV, imgHSV, Size(9, 9), 2, 2 );
  Mat imgThresholded;

   inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

  //morphological opening (remove small objects from the foreground)
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)) );
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)) );

erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)) );
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)) );

   //morphological closing (fill small holes in the foreground)
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
vector<Vec3f> circles2;
 int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
Sobel( imgThresholded, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );




  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( imgThresholded, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  //addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgThresholded );
  threshold(imgThresholded,  imgThresholded, 50, 250, CV_THRESH_BINARY);
//Canny(imgThresholded,  imgThresholded, 20, 20*2, 3 );
dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );

HoughCircles( imgThresholded,circles2, CV_HOUGH_GRADIENT, 1, imgThresholded.rows/32, xis,yip, ze, aga);

//HoughCircles( grayFrames, circles2, CV_HOUGH_GRADIENT, 1, grayFrames.rows/8, 200, 100, 0, 0 );

  for( size_t i = 0; i < circles2.size(); i++ )
  {
      Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]+400));
      int radius = cvRound(circles2[i][2]);
      // circle center
      circle( imgOriginal, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( imgOriginal, center, radius, Scalar(100,100,255), 3, 8, 0 );
   }

   //imshow("Thresholded Image", imgThresholded); //show the thresholded image
  imshow("Original", imgOriginal); //show the original image

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
       {
            cout << "esc key is pressed by user" << endl;
            break;
       }
    }

   return 0;
}
