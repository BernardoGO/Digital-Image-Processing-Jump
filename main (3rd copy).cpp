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

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;
Point pontos[50] ;
int p_pos = 0;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == EVENT_LBUTTONDOWN )
     {
          cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
          pontos[p_pos] = Point(x,y);
          p_pos++;
     }
     else if  ( event == EVENT_RBUTTONDOWN )
     {
          cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
     else if  ( event == EVENT_MBUTTONDOWN )
     {
          cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
     else if ( event == EVENT_MOUSEMOVE )
     {
          cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

     }
}

void my_mouse_callback( int event, int x, int y, int flags, Mat param );

CvRect box;
bool drawing_box = false;

void draw_box( IplImage* img, CvRect rect ){
	cvRectangle( img, cvPoint(box.x, box.y), cvPoint(box.x+box.width,box.y+box.height),
				cvScalar(0xff,0x00,0x00) );
}

// Implement mouse callback
void my_mouse_callback( int event, int x, int y, int flags, void* param ){
	IplImage* image = (IplImage*) param;

	switch( event ){
		case CV_EVENT_MOUSEMOVE:
			if( drawing_box ){
				box.width = x-box.x;
				box.height = y-box.y;
			}
			break;

		case CV_EVENT_LBUTTONDOWN:
			drawing_box = true;
			box = cvRect( x, y, 0, 0 );
			break;

		case CV_EVENT_LBUTTONUP:
			drawing_box = false;
			if( box.width < 0 ){
				box.x += box.width;
				box.width *= -1;
			}
			if( box.height < 0 ){
				box.y += box.height;
				box.height *= -1;
			}
			draw_box( image, box );
			break;
	}
}

#define MAX_COUNT 500
char rawWindow[] = "Raw Video";
char opticalFlowWindow[] = "Optical Flow Window";
char imageFileName[32];
long imageIndex = 0;
char keyPressed;

int main() {
const string& filename = "v2.avi";
 VideoCapture cap(filename);
CvCapture *capture = cvCaptureFromAVI("v7.avi");

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
Vec4i CacheLine = NULL;
Vec4i regua = NULL;
int timesFound = 0;
float reguaVals0[2000];
float reguaVals1[2000];
float reguaVals2[2000];
float reguaVals3[2000];
vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
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
imgHSV = imgHSV(Rect(0,400,1280,250));

  Mat imgThresholded;
    Mat imgLines;
cvtColor(imgOriginal, imgLines, COLOR_BGR2GRAY);
//inRange(imgLines, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgLines);
GaussianBlur( imgLines, imgLines, Size(11, 11), 3, 2 );
medianBlur(imgLines, imgLines, 3);
threshold(imgLines,  imgLines, 180, 250, CV_THRESH_BINARY);
imgLines = imgLines(Rect(550,550,300,170));
   inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
 erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)) );
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)) );

erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)) );
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)) );

   //morphological closing (fill small holes in the foreground)
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
threshold(imgThresholded,  imgThresholded, 50, 250, CV_THRESH_BINARY);

vector<Vec3f> circles2;
 int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

HoughCircles( imgThresholded,circles2, CV_HOUGH_GRADIENT, 1, imgThresholded.rows/32, xis,yip, ze, aga);

  for( size_t i = 0; i < circles2.size(); i++ )
  {
      Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]+400));
      int radius = cvRound(circles2[i][2]);
      // circle center
      circle( imgOriginal, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( imgOriginal, center, radius, Scalar(100,100,255), 3, 8, 0 );
   }


Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;


Canny(imgLines, imgLines, 30, 100, 3);
//imgLines = 255 - imgLines;
dilate( imgLines, imgLines, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
//GaussianBlur( imgLines, imgLines, Size(3, 3), 3, 2 );
//medianBlur(imgLines, imgLines, 3);
vector<Vec4i> lines;

if(framenum < 10)
{
    findContours( imgLines, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
}
vector<RotatedRect> minRect( contours.size() );
Mat drawing = Mat::zeros( imgLines.size(), CV_8UC3 );
RNG rng(12345);
cout << "ok";
for( int i = 0; i < contours.size(); i++ )
     { minRect[i] = minAreaRect( Mat(contours[i]) );

     }

//double dfd = 0;
Point local;

int maxX = 999;
Point2f rect_pointsT[4];

Point um;
Point dois;

  for( int i = 0; i< contours.size(); i++ )
     {

       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       //drawContours( imgOriginal, contours, i, color, 2, 8, hierarchy, 0, Point(550,550));
       // rotated rectangle
       Point2f rect_points[4]; minRect[i].points( rect_points );
       for( int j = 0; j < 4; j++ )
       {
          line( imgOriginal, Point(rect_points[j].x+550, rect_points[j].y+550), Point(rect_points[(j+1)%4].x+550, rect_points[(j+1)%4].y+550), color, 1, 8 );

            if(rect_points[j].y+550 <= maxX)
            {
                maxX = rect_points[j].y+550;
                //rect_pointsT = rect_points;
                um = Point(rect_points[j].x+550, rect_points[j].y+550);
                dois = Point(rect_points[(j+1)%4].x+550, rect_points[(j+1)%4].y+550);
            }
        }
     }



       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       //drawContours( imgOriginal, contours, i, color, 2, 8, hierarchy, 0, Point(550,550));
       // rotated rectangle
       //Point2f rect_points[4]; minRect[i].points( rect_points );
       //for( int j = 0; j < 4; j++ )
       //{
          //line( imgOriginal, um, dois, color, 1, 8 );

        double dfd = sqrt(pow(um.x - dois.x, 2) + pow(um.y - dois.y, 2));
/*
        if(xxd >= dfd)
        {
            dfd = xxd;
            local = Point(rect_points[j].x+555, rect_points[j].y+555);
            }
            }
*/



        string xsd;
        std::ostringstream sstream;
            std::ostringstream sstream2;
            sstream << dfd;
            std::string varAsString = sstream.str();
            sstream2 << cvRound(((dfd)) ) ;
            std::string cmte = sstream2.str();
            string text =   "30cm - " + varAsString + "px";
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.4;
        int thickness = 1;
        cv::Point textOrg(um.x - 5, um.y -5);
        cv::putText(imgOriginal, text, textOrg, fontFace, fontScale,  cv::Scalar(0,0,0250), thickness,8);




/*
  HoughLinesP(imgLines, lines, 2, CV_PI/300, 100, 50, 30 );


for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = NULL;
    //int i = 0;
    //cout << lines.size();
    if(regua[0] == 0) regua = lines[i];
        l = lines[i];
    double dfd = sqrt(pow(Point(l[0], l[1]).x - Point(l[2], l[3]).x, 2) + pow(Point(l[0], l[1]).y - Point(l[2], l[3]).y, 2));
    if(dfd > 60 && dfd < 90 && lines[i][0] != 0)
    {
    //cout << dfd;
        //if(i == 0) regua = l;
        timesFound++;
        if(l[1] < regua[1]) regua = l;
        reguaVals0[timesFound-1] = l[0];
        reguaVals1[timesFound-1] = l[1];
        reguaVals2[timesFound-1] = l[2];
        reguaVals3[timesFound-1] = l[3];

        line( imgOriginal, Point(l[0]+550, l[1]+550), Point(l[2]+550, l[3]+550), Scalar(0,0,255), 3, CV_AA);
    }
  }
Vec4i regua1 = NULL;

    for( size_t i = 0; i < timesFound; i++ )
      {

        regua1[0] += reguaVals0[0];
        regua1[1] += reguaVals1[0];
        regua1[2] += reguaVals2[0];
        regua1[3] += reguaVals3[0];

      }

        //CacheLine = lines[i];
        line( imgOriginal, Point((regua[0])+550, (regua[1])+550), Point((regua[2])+550, (regua[3])+550), Scalar(0,255,0), 3, CV_AA);
*/
     //double dfd = sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));
     //if(dfd > 35 && dfd < 50)
/*
  for( size_t i = 0; i < circles2.size(); i++ )
  {
      Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]+550));
      int radius = cvRound(circles2[i][2]);
      // circle center
      circle( imgOriginal, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( imgOriginal, center, radius, Scalar(100,100,255), 3, 8, 0 );
   }
*/
for(int uiiiii = 1; uiiiii < p_pos; uiiiii++)
   {
        cv::line(imgLines, pontos[uiiiii-1], pontos[uiiiii], cv::Scalar(0,255,0), 3, 4);

   }


  imshow("Original", imgOriginal); //show the original image
   imshow("Lines", imgLines);
   imshow("Lines", imgThresholded);
cv::setMouseCallback("Original", CallBackFunc, NULL);
        if (waitKey(1) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
       {
            cout << "esc key is pressed by user" << endl;
            break;
       }
    }

   return 0;
}
