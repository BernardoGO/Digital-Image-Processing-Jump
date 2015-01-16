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

using namespace cv;
using namespace std;



#define MAX_COUNT 500

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



int main()
{

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

double reguaSize = 0;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
bool isVertical = false;
RNG rng(12345);

void findRegua(Mat img, int framenum)
{
    Mat imgLines;
    Mat imgOriginal = img;
        cvtColor(imgOriginal, imgLines, COLOR_BGR2GRAY);


        GaussianBlur( imgLines, imgLines, Size(11, 11), 3, 2 );
        medianBlur(imgLines, imgLines, 3);
        threshold(imgLines,  imgLines, 180, 250, CV_THRESH_BINARY);
        imgLines = imgLines(Rect(550,550,300,170));





        Canny(imgLines, imgLines, 30, 100, 3);

        dilate( imgLines, imgLines, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );

        vector<Vec4i> lines;

        if(framenum < 10)
        {
            findContours( imgLines, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        }
        vector<RotatedRect> minRect( contours.size() );
        Mat drawing = Mat::zeros( imgLines.size(), CV_8UC3 );


        for( int i = 0; i < contours.size(); i++ )
        {
            minRect[i] = minAreaRect( Mat(contours[i]) );

        }

        Point local;

        int maxX = 9999;
        Point2f rect_pointsT[4];

        Point um;
        Point dois;

        for( int i = 0; i< contours.size(); i++ )
        {

            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

            Point2f rect_points[4];
            minRect[i].points( rect_points );
            for( int j = 0; j < 4; j++ )
            {
                if(framenum < 10) line( imgOriginal, Point(rect_points[j].x+550, rect_points[j].y+550), Point(rect_points[(j+1)%4].x+550, rect_points[(j+1)%4].y+550), color, 1, 8 );

                if(rect_points[j].y+550 <= maxX)
                {
                    maxX = rect_points[j].y+550;

                    um = Point(rect_points[j].x+550, rect_points[j].y+550);
                    dois = Point(rect_points[(j+1)%4].x+550, rect_points[(j+1)%4].y+550);
                }
            }
        }



        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );


        reguaSize = sqrt(pow(um.x - dois.x, 2) + pow(um.y - dois.y, 2));

        if((um.x - dois.x) >  (um.y - dois.y))
        {
            //cout << "vertical";
            isVertical = true;
        }


        string xsd;
        std::ostringstream sstream;
        std::ostringstream sstream2;
        sstream << reguaSize;
        std::string varAsString = sstream.str();
        sstream2 << cvRound(((reguaSize)) ) ;
        std::string cmte = sstream2.str();
        string text =   "30cm - " + varAsString + "px";
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Point textOrg(um.x - 5, um.y -5);
        cv::putText(imgOriginal, text, textOrg, fontFace, fontScale,  cv::Scalar(250,250,0), thickness,8);

}


void execute(char* video)
{
    CvCapture *capture = cvCaptureFromAVI(video);

    clock_t begin = clock();
    int iHighH = 210;//152
    int ze = 6;
    int aga = 11;



    Point ptInicial;
    Point ptFinal;
    double maxDist = 0;


    int framenum = 0;
    bool bSuccess = false;
    Vec4i CacheLine = NULL;
    Vec4i regua = NULL;


    while (true)
    {

        Mat imgOriginal;
        IplImage* frame = NULL;
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

        Mat imgHSV;

        cvtColor(imgOriginal, imgHSV, COLOR_BGR2GRAY);


        Mat imgThresholded;

        findRegua(imgOriginal, framenum);

        int offset_x = 400;
        int offset_x_end = 650;
        int offset_y = 0;
        int offset_y_end = 1280;


        if(isVertical)
        {
            offset_x = 0;
            offset_y = 800;
            offset_y_end = 1000;
            offset_x_end = 500;
            iHighH = 100;
        }

        imgHSV = imgHSV(Rect(offset_y,offset_x,offset_y_end-offset_y,offset_x_end-offset_x));

        imgThresholded = imgHSV;
        GaussianBlur( imgThresholded, imgThresholded, Size(11, 11), 3, 2 );
        medianBlur(imgThresholded, imgThresholded, 3);

        threshold(imgThresholded,  imgThresholded, iHighH, 250, CV_THRESH_BINARY);
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)) );
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(9, 9)) );
        Canny(imgThresholded, imgThresholded, 10, 200, 3);


        vector<vector<Point> > contoursBall;
        vector<Vec4i> hierarchyBall;

        findContours( imgThresholded, contoursBall, hierarchyBall, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        vector<Point2f>center( contoursBall.size() );
        vector<float>radius( contoursBall.size() );
        vector<Rect> boundRect( contoursBall.size() );
        vector<vector<Point> > contours_poly( contoursBall.size() );

        for( int i = 0; i < contoursBall.size(); i++ )

        {
            approxPolyDP( Mat(contoursBall[i]), contours_poly[i], 3, true );
            //boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        }

        int circles = 0;

        for( int i = 0; i< contoursBall.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            //drawContours( imgOriginal, contoursBall, i, color, 2, 8, hierarchyBall, 0, Point(800,0) );
            double area = cv::contourArea(contoursBall[i]);
            cv::Rect r = cv::boundingRect(contoursBall[i]);
            int radius2 = r.width / 2;

            if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
                    std::abs(1 - (area / (CV_PI * std::pow(radius2, 2)))) <= 0.2)
            {
                circles++;
                if(circles >= 2) break;
                if(radius[i] >= ze && radius[i] <= aga)
                {
                    circle( imgOriginal, Point(center[i].x+offset_y, center[i].y+offset_x), (int)radius[i], color, 2, 8, 0 );

                    if(framenum < 10)
                    {
                        ptInicial = Point(center[i].x+offset_y, center[i].y+offset_x);
                    }
                    else
                    {
                        Point acg = Point(center[i].x+offset_y, center[i].y+offset_x);
                        double dfd = (sqrt(pow(ptInicial.x - acg.x, 2) + pow(ptInicial.y - acg.y, 2)));
                        if(acg.y > ptInicial.y && isVertical)
                        {
                            ptInicial = acg;
                        }
                        if(dfd >= maxDist)
                        {
                            ptFinal = acg;
                            maxDist = dfd;
                            //cout << maxDist;
                        }
                    }

                }
            }

        }

        if(maxDist != 0)
        {
	    line( imgOriginal, ptInicial, ptFinal, Scalar(255,0,0), 1, 8 );
            circle( imgOriginal, ptFinal, 2, Scalar(255,0,0), 2, 8, 0 );
	    circle( imgOriginal, ptInicial, 2, Scalar(155,0,0), 2, 8, 0 );

            string xsd;
            std::ostringstream sstream;
            std::ostringstream sstream2;
            sstream << maxDist;
            std::string varAsString = sstream.str();
            sstream2 << cvRound(((maxDist/reguaSize))*30 ) ;
            std::string cmte = sstream2.str();
            string text =   cmte + "cm - " + varAsString + "px";
            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 2;
            cv::Point textOrg(ptFinal.x - 5, ptFinal.y -5);
            cv::putText(imgOriginal, text, textOrg, fontFace, fontScale,  cv::Scalar(255,180,180), thickness,8);
        }


        imshow("Original", imgOriginal); //show the original image
        //imshow("Lines", imgLines);
        //imshow("Bola", imgThresholded);
        oldImage = imgOriginal;

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
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
