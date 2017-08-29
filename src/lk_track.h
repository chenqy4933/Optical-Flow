#ifndef LK_TRACK_H
#define LK_TRACK_H

#include <opencv2/highgui/highgui.hpp>
#include<iostream>
#include<stdio.h>
#include<vector>

class LK_track
{
public: 
    LK_track();
    LK_track(cv::Mat& preIm,cv::Mat& preIm,unsigned int level_,unsigned int subsampl_);
    int Init(cv::Mat& preIm,cv::Mat& preIm,unsigned int level_,unsigned int subsampl_);
    int Computer(std::vector<cv::Point2d>& Piont_pre,std::vector<cv::Point2d>& Piont_new);
    int generate_pyramid();   //computer pyramid
    int generate_grad();      //computer grad

    unsigned int window=5;
    unsigned int window_hf=5>>1;
    unsigned int level=2;
    unsigned int subsampl=2;
    unsigned int max_iter=20;
    unsigned int max_distance=50;
    unsigned int min_dis_iter=0.03;

    std::vector<cv::Mat> Pyramid_pre;
    std::vector<cv::Mat> Pyramid_next;

    std::vector<cv::Mat> grad_next;
    cv::Mat preMat;
    cv::Mat nextMat;
    float guiss_kernel[9]={1/16,1/8,1/16,1/8,1/4,1/8,1/16,1/8,1/16};
};

#endif // LK_TRACK_H
