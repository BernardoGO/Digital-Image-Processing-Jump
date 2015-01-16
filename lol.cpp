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



const int HORIZONTAL_BORDER_CROP = 20;

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; 
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
	// "+"
	friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
	}
	//"-"
	friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
	}
	//"*"
	friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
	}
	//"/"
	friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
	}
	//"="
	Trajectory operator =(const Trajectory &rx){
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x,y,a);
	}

    double x;
    double y;
    double a; // angle
};
//



#define MAX_COUNT 500

char imageFileName[32];
long imageIndex = 0;
char keyPressed;
void execute(char* video);
void subtraction(char* video);
Mat oldImage;
int vid = 0;
int met = 0;
void runVideo()
{
	if(met==0){
	    	if(!vid){execute("v2.avi");}
	    	else if(vid == 1){execute("v6.avi");}
    		else{execute("v7.avi");}
	}
	else if(met==1){
		if(!vid){subtraction("v2.avi");}
	    	else if(vid == 1){subtraction("v6.avi");}
    		else{subtraction("v7.avi");}
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
    cvCreateTrackbar("Video", "Control", &vid, 2);
    cvCreateTrackbar("method","Control", &met, 1);
    while(1)
    {
        //imshow("Control", imgOriginal);
        cv::setMouseCallback("Control", CallBackFunc, NULL);
        Mat img = imread("play.png", CV_LOAD_IMAGE_COLOR);
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


    int framenum = 0;
    bool bSuccess = false;
    Vec4i CacheLine = NULL;
    Vec4i regua = NULL;

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
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
        Mat imgLines;
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
        RNG rng(12345);

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

void subtraction( char* video)
{	    
	
 	Mat fgMaskMOG; //fg mask generated by MOG method
 	BackgroundSubtractorMOG pMOG; // history is an int, distance_threshold is an int (usually set to 16), shadow_detection is a bool
    //namedWindow("Frame",CV_WINDOW_AUTOSIZE);
    //	namedWindow("FG Mask MOG",CV_WINDOW_AUTOSIZE);
	

	// For further analysis
	//ofstream out_transform("prev_to_cur_transformation.txt");
	//ofstream out_trajectory("trajectory.txt");
	//ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
	//ofstream out_new_transform("new_prev_to_cur_transformation.txt");

	VideoCapture cap(video);
	assert(cap.isOpened());

	Mat cur, cur_grey, conv;
	Mat prev, prev_grey;

	cap >> prev;//get the first frame.ch
	cvtColor(prev, prev_grey, COLOR_BGR2GRAY);
	
	// Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
	vector <TransformParam> prev_to_cur_transform; // previous to current
	// Accumulated frame to frame transform
	double a = 0;
	double x = 0;
	double y = 0;
	// Step 2 - Accumulate the transformations to get the image trajectory
	vector <Trajectory> trajectory; // trajectory at all frames
	//
	// Step 3 - Smooth out the trajectory using an averaging window
	vector <Trajectory> smoothed_trajectory; // trajectory at all frames
	Trajectory X;//posteriori state estimate
	Trajectory	X_;//priori estimate
	Trajectory P;// posteriori estimate error covariance
	Trajectory P_;// priori estimate error covariance
	Trajectory K;//gain
	Trajectory	z;//actual measurement
	double pstd = 4e-3;//can be changed
	double cstd = 0.25;//can be changed
	Trajectory Q(pstd,pstd,pstd);// process noise covariance
	Trajectory R(cstd,cstd,cstd);// measurement noise covariance 
	// Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
	vector <TransformParam> new_prev_to_cur_transform;
	//
	// Step 5 - Apply the new transformation to the video
	//cap.set(CV_CAP_PROP_POS_FRAMES, 0);
	Mat T(2,3,CV_64F);

	int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
	VideoWriter outputVideo; 
	outputVideo.open("compare.avi" , CV_FOURCC('X','V','I','D'), 24,cvSize(cur.rows, cur.cols*2+10), true);  
	//
	int k=1;
	int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	///contornos AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!AQUI!!!!	
	int thresh = 100;
	int max_thresh = 255;
	RNG rng(12345);
	Mat canny_output;
  	vector<vector<Point> > contours, contoursR;
  	vector<Vec4i> hierarchy, hierarchyR;
	vector<Point> biggestContour;
	Moments momento;
	Rect mais_baixo;
	Rect mais_esquerda;
	bool isVertical = false;
	int iHighH = 210;
	Point baixo, alto;
    	double reguaSize = 0;
	double ult_desloc=0;
	double deslocamento; //mais baixo - mais esquerda	
	//vector<Point> approx;
	///fim contornos

	
	Mat last_T;
	Mat prev_grey_,cur_grey_;
	Mat edges, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj; 
	while(true) {
		cap >> cur;
		if(cur.data == NULL) {
			break;
		}

		cvtColor(cur, cur_grey, CV_BGR2GRAY);

		// vector from prev to cur
		vector <Point2f> prev_corner, cur_corner;
		vector <Point2f> prev_corner2, cur_corner2;
		vector <uchar> status;
		vector <float> err;

		goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
		calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

		// weed out bad matches
		for(size_t i=0; i < status.size(); i++) {
			if(status[i]) {
				prev_corner2.push_back(prev_corner[i]);
				cur_corner2.push_back(cur_corner[i]);
			}
		}

		// translation + rotation only
		Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

		// in rare cases no transform is found. We'll just use the last known good transform.
		if(T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		// decompose T
		double dx = T.at<double>(0,2);
		double dy = T.at<double>(1,2);
		double da = atan2(T.at<double>(1,0), T.at<double>(0,0));
		//
		//prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

		//out_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		// Accumulated frame to frame transform
		x += dx;
		y += dy;
		a += da;
		//trajectory.push_back(Trajectory(x,y,a));
		//
		//out_trajectory << k << " " << x << " " << y << " " << a << endl;
		//
		z = Trajectory(x,y,a);
		//
		if(k==1){
			// intial guesses
			X = Trajectory(0,0,0); //Initial estimate,  set 0
			P =Trajectory(1,1,1); //set error variance,set 1
		}
		else
		{
			//time 
			X_ = X; //X_(k) = X(k-1);
			P_ = P+Q; //P_(k) = P(k-1)+Q;
			// measurement
			K = P_/( P_+R ); //gain;K(k) = P_(k)/( P_(k)+R );
			X = X_+K*(z-X_); //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k)); 
			P = (Trajectory(1,1,1)-K)*P_; //P(k) = (1-K(k))*P_(k);
		}
		//smoothed_trajectory.push_back(X);
		//out_smoothed_trajectory << k << " " << X.x << " " << X.y << " " << X.a << endl;
		//-
		// target - current
		double diff_x = X.x - x;//
		double diff_y = X.y - y;
		double diff_a = X.a - a;

		dx = dx + diff_x;
		dy = dy + diff_y;
		da = da + diff_a;

		//new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
		//
		//out_new_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		T.at<double>(0,0) = cos(da);
		T.at<double>(0,1) = -sin(da);
		T.at<double>(1,0) = sin(da);
		T.at<double>(1,1) = cos(da);

		T.at<double>(0,2) = dx;
		T.at<double>(1,2) = dy;
		Mat save = cur;
		cvtColor(cur, cur, CV_BGR2GRAY);
		
		//corta borda
		int offset_x = 400;
	        int offset_x_end = 600;
	        int offset_y = 300;
	        int offset_y_end = 1000;
       		if(isVertical)
        	{
          	 offset_x = 0;
          	 offset_y = 800;
          	 offset_y_end = 1000;
           	 offset_x_end = 250;
           	 iHighH = 100;
        	}
		
			Point local;
		int maxX = 9999;
		Point2f rect_pointsT[4];
		Point um;
		Point dois;

///
		Mat imgHSV;

		cvtColor(save, imgHSV, COLOR_BGR2GRAY);
		Mat imgThresholded;
        	Mat imgLines;
		cvtColor(save, imgLines, COLOR_BGR2GRAY);
		GaussianBlur( imgLines, imgLines, Size(11, 11), 3, 2 );
		medianBlur(imgLines, imgLines, 3);
		threshold(imgLines,  imgLines, 180, 250, CV_THRESH_BINARY);
		imgLines = imgLines(Rect(550,550,300,170));
		Canny(imgLines, imgLines, 30, 100, 3);
		dilate( imgLines, imgLines, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
		vector<Vec4i> lines;
		if(k < 4)
		{
		  
			findContours( imgLines, contoursR, hierarchyR, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        	
		vector<RotatedRect> minRect( contoursR.size() );
		
		RNG rngR(12345);

		for( int i = 0; i < contoursR.size(); i++ )
		{
			minRect[i] = minAreaRect( Mat(contoursR[i]) );
		}
	

		for( int i = 0; i< contoursR.size(); i++ )
		{
			Scalar color2 = Scalar( rngR.uniform(0, 255), rngR.uniform(0,255), rngR.uniform(0,255) );
		Point2f rect_pointsR[4];
		minRect[i].points( rect_pointsR );
            	for( int j = 0; j < 4; j++ )
            	{
                if(k < 10) line( save, Point(rect_pointsR[j].x+550, rect_pointsR[j].y+550), Point(rect_pointsR[(j+1)%4].x+550, 			rect_pointsR[(j+1)%4].y+550), color2, 1, 8 );

                if(rect_pointsR[j].y+550 <= maxX)
                {
                    maxX = rect_pointsR[j].y+550;

                    um = Point(rect_pointsR[j].x+550, rect_pointsR[j].y+550);
                    dois = Point(rect_pointsR[(j+1)%4].x+550, rect_pointsR[(j+1)%4].y+550);
                }
            }
        }
        reguaSize = sqrt(pow(um.x - dois.x, 2) + pow(um.y - dois.y, 2));

        if((um.x - dois.x) >  (um.y - dois.y))
        {
            //cout << "vertical";
            isVertical = 1;
        }
		}
///
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
        cv::putText(save, text, textOrg, fontFace, fontScale,  cv::Scalar(250,250,0), thickness,8);
	//cout << "kaka";

cur = cur(Rect(offset_y,offset_x,offset_y_end-offset_y,offset_x_end-offset_x));

//cout << isVertical<< endl;
		//cur = cur(Rect(0,400,1280,650));
        	pMOG.operator()(cur, fgMaskMOG);
		GaussianBlur( fgMaskMOG, fgMaskMOG, Size(15, 15), 6, 6 );
        	erode(fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_RECT, Size(1, 1)) );
        	dilate( fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_RECT, Size(1, 1)) );
        	erode(fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_RECT, Size(1, 1)) );
        	dilate( fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_RECT, Size(1, 1)) );
        	erode(fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_RECT, Size(1, 1)) );
       		dilate( fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_RECT, Size(1, 1)) );
        //	erode(fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)) );
        //	dilate( fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)) );
        	erode(fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)) );
        	dilate( fgMaskMOG, fgMaskMOG, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)) );
		threshold(fgMaskMOG,fgMaskMOG,150,255,CV_THRESH_BINARY);      		
		//bordas
		Canny( fgMaskMOG, canny_output, thresh, thresh*2, 3 );
		//vetor de contornos		
		findContours( canny_output, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );
		//imshow("stuff",canny_output);
		  /// Approximate contours to polygons + get bounding rects and circles
  		vector<vector<Point> > contours_poly( contours.size() );
  		vector<Rect> boundRect( contours.size() );
  		vector<Point2f>center( contours.size() );
  		vector<float>radius( contours.size() );

 		for( int lol = 0; lol < contours.size(); lol++ )
    		{
			approxPolyDP( Mat(contours[lol]), contours_poly[lol], 2, true );
 			boundRect[lol] = boundingRect( Mat(contours_poly[lol]) );
  			
   		}
		
		for(int lol = 0; lol<boundRect.size();lol++)
		{//mais_baixo
		//mais_esquerda
		//mais baixo - mais esquerda	
		//cout <<"rectx" << boundRect[lol].x << "recty" <<boundRect[lol].y << endl;
			//pegar o inicial
			bool b=false;			
			if(k>10)
			{
			 mais_baixo=boundRect[lol];
			 mais_esquerda=boundRect[lol];

			 if(baixo.y <= (boundRect[lol].tl()).y)
			   baixo=boundRect[lol].tl();


			 alto=boundRect[lol].tl();
						
			}
			
			//mais baixo
			if(isVertical == false)
			{
				if(mais_baixo.y>boundRect[lol].y)
				  {mais_baixo=boundRect[lol];
				}
				//mais esquerda
				if(mais_esquerda.x>boundRect[lol].x)
				{mais_esquerda=boundRect[lol];
				}
				if(k>50)
				{deslocamento= sqrt(pow(mais_baixo.x - mais_esquerda.x, 2) + pow(mais_baixo.y - mais_esquerda.y, 2));
				cout << deslocamento << endl;
				}
			}
			if(isVertical==true)
			{
			//mais_esquerda funciona como se fosse um mais_acima
				if(baixo.y>boundRect[lol].y)
				{baixo=boundRect[lol].tl();
					cout << baixo << endl;
				}
				//mais esquerda --> vira mais acima
				if(alto.y>boundRect[lol].y)
				{alto=boundRect[lol].tl();
				}
			
			}
			if(k>50&&isVertical==0)
			{deslocamento= sqrt(pow(mais_baixo.x - mais_esquerda.x, 2) + pow(mais_baixo.y - mais_esquerda.y,2));
			}else if(k>50&&isVertical==1)
			{
			  deslocamento=sqrt(pow(baixo.x-alto.x,2)+pow(baixo.y-alto.y,2));
				
			}
		}	
		
		if(ult_desloc<deslocamento)
		  {ult_desloc=deslocamento;}
		Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
		//moments(, bool binaryImage=false )
		for( int contadoroContorno = 0; contadoroContorno< contours.size(); contadoroContorno++ )
     		{
       			//Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       			//drawContours( drawing, contours, contadoroContorno, color, CV_FILLED, CV_AA, hierarchy, 0, Point() );
			//rectangle( save, boundRect[contadoroContorno].tl(), boundRect[contadoroContorno].br(), color, 2, 8, 0 );
     		}	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			rectangle( save, Point(mais_esquerda.tl().x+offset_y,mais_esquerda.tl().y+offset_x), Point(mais_esquerda.br().x+offset_y,mais_esquerda.br().y+offset_x), color, 2, 8, 0 );
			rectangle( save, Point(mais_baixo.tl().x+offset_y,mais_baixo.tl().y+offset_x), Point(mais_baixo.br().x+offset_y,mais_baixo.br().y+offset_x), color, 2, 8, 0 );	


//
if(ult_desloc>0)
{
            string xsd;
            std::ostringstream sstream;
            std::ostringstream sstream2;
            sstream << ult_desloc;
            std::string varAsString = sstream.str();
            sstream2 << cvRound(((ult_desloc/reguaSize))*30 );
	    cout << "desloc" << ult_desloc<< "regua"<< reguaSize<< endl; 
            std::string cmte = sstream2.str();
            string text =   cmte + "cm - " + varAsString + "px";
            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 2;
	//cout << text <<endl;
            cv::Point textOrg(mais_esquerda.tl().x +offset_y, mais_esquerda.tl().y + offset_x);
            cv::putText(save, text, textOrg, fontFace, fontScale,  cv::Scalar(20,20,20), thickness,8);

}

		imshow("Subtracao",save);

		waitKey(10);
		//
		prev = cur.clone();//cur.copyTo(prev);
		cur_grey.copyTo(prev_grey);
		 
		cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
				
		k++;

	 if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            string text =   "esc key pressed by the user";
            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 2;
            int thickness = 2;
            cv::Point textOrg(200, 600);
            cv::putText(save, text, textOrg, fontFace, fontScale,  cv::Scalar(0,0,120), thickness,8);
            imshow("Original", save);
            break;
        }

	}
	//return 0;
}

