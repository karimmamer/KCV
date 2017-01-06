#ifndef KARIMCV_H
#define KARIMCV_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>

cv::Mat edge_detection(cv::Mat img, int kernel_w = 5, int kernel_h = 5, int sigma_x = 0, int sigma_y = 0, int low_threshold = 75, int high_threshold = 200, int neighbour_size = 1); //implementation of canny edge detection algorithm using opencv functions

cv::Mat corner_detection(cv::Mat img, float threshold = 75, int kernel_w = 5, int kernel_h = 5, int sigma_x = 0, int sigma_y = 0, float alpha = 0.06, int neighbour_size = 5);

cv::Mat hough_line_detection(cv::Mat img, float threshold = 20, int neighbour_size = 1, int theta_step = 5, int p_step = 5);

cv::Mat hough_circle_detection(cv::Mat img, float threshold = 300, int neighbour_size = 5, int rad_step = 4, int center_step = 4);

cv::Mat optical_flow_lucas_kanade(cv::Mat img1, cv::Mat img2, int w = 1);

cv::Mat optical_flow_horn_schunk(cv::Mat img1, cv::Mat img2, float lambda = 1.0, int iter = 5);

void draw_line(cv::Mat & img, int theta, int p);

void draw_circle(cv::Mat & img, int a, int b, int r);

int approximate(float x, int step);

cv::Mat derivative_x(cv::Mat img1, cv::Mat img2);

cv::Mat derivative_y(cv::Mat img1, cv::Mat img2);

cv::Mat derivative_t(cv::Mat img1, cv::Mat img2);

cv::Mat color_coding(cv::Mat x, cv::Mat y);

class Image
{
    cv::Mat img;
    std::string name;
    
public:

    Image(std::string fname);
    void display(std::string preprocessing);
};

class Video
{
    cv::VideoCapture cap;
    std::string name;
    
public:

    Video(std::string fname);
    void display(std::string preprocessing);
};

#endif
