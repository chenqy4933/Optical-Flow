#include<stdio.h>
#include<iostream>
#include"lk_track.h"
#include<opencv2/highgui/highgui.hpp>
#include "cv.h"
#include "highgui.h"

using namespace std;

int main()
{
    cv::Mat prevGray = cv::imread("basketball1.png", CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat gray = cv::imread("basketball2.png", CV_LOAD_IMAGE_GRAYSCALE);
        if (NULL == prevGray.data || NULL == gray.data)
        {
            cout << "failed to load images" << endl;
            return 0;
        }
        //FAST特征点检测
        std::vector<cv::KeyPoint> vecPreKeyPoints;
        //Ptr<FastFeatureDetector> fast(new FastFeatureDetector);
        //fast->setThreshold(40);
        //fast->detect(prevGray, vecPreKeyPoints);
        //FastFeatureDetector fast(40);
        //fast.detect(prevGray, vecPreKeyPoints);
        FAST(prevGray, vecPreKeyPoints, 40);

        std::cout<<"Keypoint size is "<<vecPreKeyPoints.size()<<std::endl;
        vector<cv::Point2f> vecPrePoints;
        vecPrePoints.reserve(vecPreKeyPoints.size());

        for (int i = 0; i < vecPreKeyPoints.size(); i++)
        {
            vecPrePoints.push_back(vecPreKeyPoints[i].pt);
        }

        //光流跟踪
        if (!vecPreKeyPoints.empty())
        {
            vector<cv::Point2f> vecCurrentPoints;
            vector<uchar> status;
            vector<float> err;
            TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
            Size winSize(31, 31);

            int start=clock();
            for(int i=0;i<100;i++)
                calcOpticalFlowPyrLK(prevGray, gray, vecPrePoints, vecCurrentPoints, status, err, winSize,3, termcrit, 0, 0.001);
            int stop=clock();
            std::cout<<"Opencv LK time is "<<(double)(stop-start)/100/1000000<<std::endl;
            //显示跟踪结果

            cv::Mat objImgShow;
            cv::cvtColor(gray, objImgShow, CV_GRAY2BGR);
            for (size_t i = 0; i < vecPreKeyPoints.size(); i++)
            {

                    const int RADIUS = 2;
                    cv::circle(objImgShow, vecCurrentPoints[i], RADIUS, CV_RGB(255, 0, 0), CV_FILLED);
                    cv::line(objImgShow, vecPrePoints[i], vecCurrentPoints[i], CV_RGB(0, 255, 0));

            }
            cv::imshow("Optical Flow OpenCV", objImgShow);
        }
        //int number=vecPrePoints.size();
        int width=prevGray.cols;
        int heigh=prevGray.rows;
        std::cout<<width<<std::endl;
        std::cout<<heigh<<std::endl;

        vector<cv::Point2f> vecNextPoints(vecPrePoints.size());
        LK_track track(prevGray,gray,2,4);
        track.Computer(vecPrePoints,vecNextPoints);

        cv::Mat objImgShow2;
        cv::cvtColor(gray, objImgShow2, CV_GRAY2BGR);
        for (size_t i = 0; i < vecPrePoints.size(); i++)
        {

                const int RADIUS = 2;
                cv::circle(objImgShow, vecNextPoints[i], RADIUS, CV_RGB(255, 0, 0), CV_FILLED);
                cv::line(objImgShow, vecPrePoints[i], vecNextPoints[i], CV_RGB(0, 255, 0));

        }
        cv::imshow("Optical Flow Myflow", objImgShow2);

        cv::waitKey();
        return 0;
	return 0;
}
