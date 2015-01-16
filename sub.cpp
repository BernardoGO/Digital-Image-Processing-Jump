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
#include <ctime>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

char imageFileName[32];
long imageIndex = 0;
char keyPressed;
void execute(char* video);
Mat oldImage;
int vid = 0;

void runVideo()
{
    if(!vid)
    {
        execute("v2.avi");
    }
    else if(vid == 1)
    {
        execute("v6.avi");
    }
    else
    {
        execute("v7.avi");
    }
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        //cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        runVideo();
	
    }
    else if  ( event == EVENT_RBUTTONDOWN )
    {
        //cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if  ( event == EVENT_MBUTTONDOWN )
    {
        //cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if ( event == EVENT_MOUSEMOVE )
    {
        //cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

    }
}

cv::BackgroundSubtractorMOG2 bg;
cv::Mat back;
cv::Mat fore;

std::vector<std::vector<cv::Point> > contours;

int main()
{
  //bg.nmixtures = 3;
  // bg.bShadowDetection = false;
  namedWindow("Control", CV_WINDOW_AUTOSIZE);
  cvCreateTrackbar("Video", "Control", &vid, 2); //Hue (0 - 179)

  while(1)
    {
      //imshow("Control", imgOriginal);
      cv::setMouseCallback("Control", CallBackFunc, NULL);
      Mat img = imread("play.png", CV_LOAD_IMAGE_COLOR);
      //cvRectangle(img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness=1, int line_type=8, int shift=0 )
      imshow("Control", img);
      if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
	  cout << "esc key is pressed by user" << endl;
	  break;
        }
    }
  //execute("v2.avi");
  return 0;
}

void execute(char* video)
{
    CvCapture *capture = cvCaptureFromAVI(video);
    clock_t begin = clock();
    int iHighH = 210;//152
    int ze = 6;
    int aga = 11;

    bool isVertical = false;
    double reguaSize = 0;
    Point ptInicial;
    Point ptFinal;
    double maxDist = 0;
    Mat imgSub;

    int framenum = 0;
    bool bSuccess = false;
    Vec4i CacheLine = NULL;
    Vec4i regua = NULL;

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
Mat imgOriginal;
IplImage* frame = NULL;
    frame = cvQueryFrame(capture);
        imgOriginal = frame;
	if(framenum == 0)
	  {
	    imgSub = imgOriginal;
	    
	  }

    while (true)
    {

      
        
        frame = cvQueryFrame(capture);
        imgOriginal = frame;

        framenum++;

        if (!frame)
        {
            //cout << "Cannot read a frame from video stream" << endl;

            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            cout << "Time Elapsed: " << elapsed_secs << " seconds" << endl;
            begin = clock();
            string xsd;
            std::ostringstream sstream;
            std::ostringstream sstream2;
            sstream << reguaSize;
            std::string varAsString = sstream.str();
            sstream2 << (((elapsed_secs)) ) ;
            std::string cmte = sstream2.str();
            string text =   "Time Elapsed: " + cmte + " seconds";
            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 2;
            int thickness = 2;
            cv::Point textOrg(200, 600);
            cv::putText(oldImage, text, textOrg, fontFace, fontScale,  cv::Scalar(0,0,120), thickness,8);


            imshow("Original", oldImage);

            if(false)
            {
                cvSetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO , 0);

            }
            break;
        }
	
	//GaussianBlur( imgOriginal, imgOriginal, Size(11, 11), 3, 2 );
	// medianBlur(imgOriginal, imgOriginal, 3);

	//cv::imshow("Frame",fore);
        //cv::imshow("Background",back);
	
       	imgSub = imgSub - imgOriginal;

	imshow("Original2", imgSub); //show the original image
        //imshow("Lines", imgLines);
        //imshow("Lines", imgThresholded);
        oldImage = imgOriginal;

        if (waitKey(1) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            string text =   "esc key pressed by the user";
            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 2;
            int thickness = 2;
            cv::Point textOrg(200, 600);
            cv::putText(imgOriginal, text, textOrg, fontFace, fontScale,  cv::Scalar(0,0,120), thickness,8);
            imshow("Original", imgOriginal);
            break;
        }
    }


}
