#include<opencv2/highgui/highgui.hpp>
#include "cv.h"
#include "highgui.h"

#include<stdio.h>
#include<iostream>
#include<vector>
#include "lk_track.h"

LK_track::LK_track()
{
    level=0;
    subsampl=0;
    preMat=NULL;
    nextMat=NULL;
    Pyramid_pre.resize(0);
    Pyramid_next.resize(0);
    grad_pre.resize(0);
    grad_next.resize(0);
}

LK_track::LK_track(cv::Mat& preIm,cv::Mat& nextIm,unsigned int level_,unsigned int subsampl_)
{
    level=level_;
    subsampl=subsampl_;
    preMat=preIm;
    nextMat=nextIm;
    Pyramid_pre.resize(level);
    Pyramid_next.resize(level);
    grad_pre.resize(level);
    grad_next.resize(level);

    generate_pyramid();
    generate_grad();


}
int LK_track::Init(cv::Mat& preIm,cv::Mat& nextIm,unsigned int level_,unsigned int subsampl_)
{
    level=level_;
    subsampl=subsampl_;
    preMat=preIm;
    nextMat=nextIm;
    Pyramid_pre.resize(level);
    Pyramid_next.resize(level);
    grad_pre.resize(level);
    grad_next.resize(level);

    generate_pyramid();
    generate_grad();
    return 1;

}
int LK_track::generate_pyramid()   //computer pyramid
{
    cv::Mat resImage;
    if(preMat.channels()!=1)
    {
        cvtColor(preMat, resImage, CV_RGB2GRAY);//把图片转化为灰度图
        Pyramid_pre[0]=resImage;
    }
    resImage.empty();
    if(nextMat.channels()!=1)
    {
        cvtColor(nextMat, resImage, CV_RGB2GRAY);//把图片转化为灰度图
        Pyramid_next[0]=resImage;
    }
    for(int lev=0;lev<level-1;lev++)
    {
        Pyramid_pre[lev+1].creat((Pyramid_pre[lev].cols-1)/subsampl,(Pyramid_pre[lev].rows-1)/subsampl,CV_8U );
        U8p source =(U8p)Pyramid_pre[lev].data;
        U8p target =(U8p)Pyramid_pre[lev+1].data;
        int col=Pyramid_pre[lev+1].cols-1;
        int row=Pyramid_pre[lev+1].rows-1;
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                float temp=0.0f;

                temp+=guiss_kernel[0]*source[(i*col*2+j)*2-2*col-1];
                temp+=guiss_kernel[1]*source[(i*col*2+j)*2-2*col];
                temp+=guiss_kernel[2]*source[(i*col*2+j)*2-2*col+1];
                temp+=guiss_kernel[3]*source[(i*col*2+j)*2-1];
                temp+=guiss_kernel[4]*source[(i*col*2+j)*2];
                temp+=guiss_kernel[5]*source[(i*col*2+j)*2+1];
                temp+=guiss_kernel[6]*source[(i*col*2+j)*2+2*col-1];
                temp+=guiss_kernel[7]*source[(i*col*2+j)*2+2*col];
                temp+=guiss_kernel[8]*source[(i*col*2+j)*2+2*col+1];

                target[i*col+j]=(U8)temp;
            }
        }
    }
    for(int lev=0;lev<level-1;lev++)
    {
        Pyramid_next[lev+1].creat((Pyramid_next[lev].cols-1)/subsampl,(Pyramid_next[lev].rows-1)/subsampl,CV_8U );
        U8p source =Pyramid_next[lev].data;
        U8p target =Pyramid_next[lev+1].data;
        int col=Pyramid_next[lev+1].cols-1;
        int row=Pyramid_next[lev+1].rows-1;
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                float temp=0.0f;

                temp+=guiss_kernel[0]*source[(i*col*2+j)*2-2*col-1];
                temp+=guiss_kernel[1]*source[(i*col*2+j)*2-2*col];
                temp+=guiss_kernel[2]*source[(i*col*2+j)*2-2*col+1];
                temp+=guiss_kernel[3]*source[(i*col*2+j)*2-1];
                temp+=guiss_kernel[4]*source[(i*col*2+j)*2];
                temp+=guiss_kernel[5]*source[(i*col*2+j)*2+1];
                temp+=guiss_kernel[6]*source[(i*col*2+j)*2+2*col-1];
                temp+=guiss_kernel[7]*source[(i*col*2+j)*2+2*col];
                temp+=guiss_kernel[8]*source[(i*col*2+j)*2+2*col+1];

                target[i*col+j]=(U8)temp;
            }
        }
    }
    return 1;
}

int LK_track::generate_grad()
{

    for(int lev=0;lev<level;lev++)
    {
        int row=Pyramid_next[lev].rows;
        int col=Pyramid_next[lev].cols;

        grad_next[lev].creat(col,row,CV_32FC2);
        U8p source=Pyramid_next[lev].data;
        float* target=grad_next[lev].data;
        for(int i=1;i<row-1;i++)
        {
            for(int j=1;j<col-1;j++)
            {
                target[(i*col+j)*2]=(source[i*col+j+1]-source[i*col+j-1])/2;  //dx
                target[(i*col+j)*2+1]=(source[(i+1)*col+j]-source[(i-1)*col+j])/2;  //dy
            }
        }
    }

}

int LK_track::Computer(std::vector<cv::Point2d>& Point_pre,std::vector<cv::Point2d>& Ponit_new)
{
    int point_size=Piont_pre.size();
    for(int i=0;i<point_size;i++)
    {
        Point_new[i]=Point_pre[i];
        for(int lev=1;lev>=0;lev--)
        {
            int row=Pyramid_next[lev].rows;\
            int col=Pyramid_next[lev].cols;
            cv::Point2d target=point_new[i];
            cv::Point2d source=Point_pre[i];
            float dis=0;
            float change=4;
            int iter=0;
            while(dis>min_dis_iter && iter<max_iter)
            {
                if(target.x-window_hf<0||target.x+window_hf>=row||
                        target.y-window_hf<0||target.y+window_hf>=col
                        ||change<min_change )
                {
                    break;
                }
                float dxdx=0,dydy=0,dxdy=0,sumdx=0,sumdy=0;
                float diff=Interpolation(Pyramid_pre[lev],source)-
                            Interpolation(Pyramid_next[lev],target);
                for(int i=-window_hf;i<=window_hf;i++)
                {
                    for(int j=-window_hf;j<=window_hf;j++)
                    {
                        cv::Point2d temp(target.x+i,target.y+j);
                        float dx=Interpolation_grad(grad_next[lev],temp,0);
                        float dy=Interpolation_grad(grad_next[lev],temp,1);
                        dxdx+=dx*dx;
                        dydy+=dy*dy;
                        dxdy+=dx*dy;
                        sumdx+=dx;
                        sumdy+=dy;
                    }
                }
                float ex=sumdx*diff;
                float ey=sumdy*diff;
                float det=1/(dxdx*dydy-dxdy*dxdy);
                float speedx = (dydy *ex  - dxdy * ey) * det;
                float speedy = (dxdx * ey - dxdy * ex) * det;
                change=speedx*speedx+speedy*speedy;
                target.x+=speedx;
                target.y+=speedy;
                dis=0;
                for(int i=-window_hf;i<=window_hf;i++)
                {
                    for(int j=-window_hf;j<=window_hf;j++)
                    {
                        cv::Point2d source_temp(source.x+i,source.y+j);
                        cv::Point2d target_temp(target.x+i,target.y+j);
                        U8 pre=Interpolation(Pyramid_pre[lev],source_temp);
                        U8 next=Interpolation(grad_next[lev],target_temp);
                        dis+=(pre-next)*(pre-next);
                    }
                }
            }
            point_new[i].x=subsampl*target.x+subsampl*target.y;
        }
    }
}
U8 LK_track::Interpolation(cv::Mat& img,cv::Point2d& position)
{
    const int FACTOR = 2048;
    const int BITS = 22;
    unsigned char* src=img.data;
    int row=img.rows;
    int col=img.cols;
    int x0, y0;
    int u, v, u_1, v_1;
    x0 = (int) (position.x);
    y0 = (int) (position.y);
    u = (x - x0) * FACTOR;
    v = (y - y0) * FACTOR;
    u_1 = FACTOR - u;
    v_1 = FACTOR - v;
    int result = (src(x0*col+y0) * u_1 + src(y0+col*(x0 + 1)) * u) * v_1
                 + (src(y0 + 1+x0*col) * u_1 + src(y0 + 1+(x0 + 1)*col) * u) * v;
    return (U8) (result >> BITS);
}

float LK_track::Interpolation_grad(cv::Mat& img,cv::Point2d& position,int flag)
{

        const int FACTOR = 2048;
        const int BITS = 22;
        float* src=img.data;
        int row=img.rows;
        int col=img.cols;
        int x0, y0;
        int u, v, u_1, v_1;
        x0 = (int) (position.x);
        y0 = (int) (position.y);
        u = (x - x0) * FACTOR;
        v = (y - y0) * FACTOR;
        u_1 = FACTOR - u;
        v_1 = FACTOR - v;
        int result = ( src((x0*col+y0)*2+flag) * u_1 + src(((x0+1)*col+y0)*2+flag) * u) * v_1
                     + (src((x0*col+y0+1)*2+falg) * u_1 + src(((x0+1)*col+y0+1)*2+falg) * u) * v;
        return result / FACTOR;

}
