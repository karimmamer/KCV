#include "karimcv.h"
using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////////////////////////////////////////
//implementation of Image class functions
///////////////////////////////////////////////////////////////////////////////////////////////////

Image::Image(string fname)
{
    img = imread(fname, CV_LOAD_IMAGE_UNCHANGED);
    if (img.empty()) //check whether the image is loaded or not
    {
      cout << "Error : Image cannot be loaded..!!" << endl;
      system("pause"); //wait for a key press
    }
    name = fname;
}

void Image::display(string preprocessing)
{
    namedWindow(name, CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
    Mat displayed_img;
    if(preprocessing == "Edge Detection")
        displayed_img = edge_detection(img);
    else if(preprocessing == "Corner Detection")
        displayed_img = corner_detection(img);
    else if(preprocessing == "Line Detection")
        displayed_img = hough_line_detection(img);
    else if(preprocessing == "Circle Detection")
        displayed_img = hough_circle_detection(img);
    else
        displayed_img = img;
    imshow(name, displayed_img); //display the image which is stored in the 'img' in the "MyWindow" window

    waitKey(0); //wait infinite time for a keypress

    destroyWindow(name); //destroy the window with the name, "MyWindow"        
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//implementation of Video class functions
///////////////////////////////////////////////////////////////////////////////////////////////////

Video::Video(string fname)
{
    bool exist = cap.open(fname);
    if (exist) //check whether the image is loaded or not
    {
      cout << "Error : Video cannot be loaded..!!" << endl;
      system("pause"); //wait for a key press
    }
    name = fname;
}

void Video::display(string preprocessing)
{
    namedWindow(name, CV_WINDOW_AUTOSIZE);
    while(1)
    {
        Mat frame;
        bool bSuccess = cap.read(frame); // read a new frame from video
        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }
        Mat displayed_frame;
        if(preprocessing == "Edge Detection")
            displayed_frame = edge_detection(frame);
        else if(preprocessing == "Corner Detection")
            displayed_frame = corner_detection(frame);
        else
            displayed_frame = frame;
        imshow(name, displayed_frame); //show the frame in "MyVideo" window
        if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl; 
            break; 
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//implementation of computer vision functions for karimcv library
///////////////////////////////////////////////////////////////////////////////////////////////////

Mat edge_detection(Mat img, int kernel_w, int kernel_h, int sigma_x, int sigma_y, int low_threshold, int high_threshold, int neighbour_size)
{
    int scale = 1, delta = 0; //constants for Sobel function
    
    Mat img_gray, img_blurred, img_grad_x, img_grad_y, img_grad_mag, img_grad_theta, temp;
    
    GaussianBlur(img, img_blurred, Size(kernel_w, kernel_h), sigma_x, sigma_y); //apply guasian filter

    if(img_blurred.channels() == 3)
        cvtColor(img_blurred, img_gray, CV_BGR2GRAY); //convert input image to gray
    else
        img_blurred.copyTo(img_gray);

    Sobel(img_gray, img_grad_x, CV_32F, 1, 0, 3, scale, delta, BORDER_DEFAULT); //apply sobel operator of degree one to direction-x
    Sobel(img_gray, img_grad_y, CV_32F, 0, 1, 3, scale, delta, BORDER_DEFAULT); //apply sobel operator of degree one to direction-y
    
    magnitude(img_grad_x, img_grad_y, img_grad_mag); //calculate gradient magnitude
    
    //calculate gradient direction
    img_grad_theta = Mat::zeros(img_grad_mag.rows, img_grad_mag.cols, CV_8U);
    for(int i = 0; i < img_grad_theta.rows; i++)
        for(int j = 0; j < img_grad_theta.cols; j++)
        {
            float angle = atan2(img_grad_y.at<float>(i, j), img_grad_x.at<float>(i, j)) * 180 / CV_PI;
            if(angle < 0.0)
            {
                angle += 180.0;
                //cout<<"y\n";
            }
            if(angle<=22.5)
                img_grad_theta.at<uchar>(i, j) = 0;
            else if(22.5<angle && angle<=67.5)
                img_grad_theta.at<uchar>(i, j) = 45;
            else if(67.5<angle && angle<=112.5)
                img_grad_theta.at<uchar>(i, j) = 90;
            else if(112.5<angle && angle<=157.5)
                img_grad_theta.at<uchar>(i, j) = 135;
            else 
                img_grad_theta.at<uchar>(i, j) = 0;
        }
    
    
    //applying non-maximum suppression
    img_grad_mag.copyTo(temp);
    int n = img_grad_mag.rows, m = img_grad_mag.cols;
    for(int i = 0; i < img_grad_mag.rows; i++)
        for(int j = 0; j < img_grad_mag.cols; j++)
        {
            int angle = img_grad_theta.at<uchar>(i, j);
            int mag = img_grad_mag.at<float>(i, j);
            float zero = 0;
            if(angle == 90)
            {
                if(i-1 >= 0 && mag <= img_grad_mag.at<float>(i-1, j))
                    temp.at<float>(i, j) = zero;
                else if(i+1 < n && mag <= img_grad_mag.at<float>(i+1, j))
                    temp.at<float>(i, j) = zero;
            }
            else if(angle == 135)
            {
                if(i-1 >= 0 && j-1 >= 0 && mag <= img_grad_mag.at<float>(i-1, j-1))
                    temp.at<float>(i, j) = zero;
                else if(i+1 < n && j+1 < m && mag <= img_grad_mag.at<float>(i+1, j+1))
                    temp.at<float>(i, j) = zero;
            }
            if(angle == 0)
            {
                if(j-1 >= 0 && mag <= img_grad_mag.at<float>(i, j-1))
                    temp.at<float>(i, j) = zero;
                else if(j+1 < n && mag <= img_grad_mag.at<float>(i, j+1))
                    temp.at<float>(i, j) = zero;
            }
            else if(angle == 45)
            {
                if(i-1 >= 0 && j+1 < m && mag <= img_grad_mag.at<float>(i-1, j+1))
                    temp.at<float>(i, j) = zero;
                else if(i+1 < n && j-1 >= 0 && mag <= img_grad_mag.at<float>(i+1, j-1))
                    temp.at<float>(i, j) = zero;
            }
        }
    temp.copyTo(img_grad_mag);

    //applying hysteresis thresholding and linking
    
    for(int i = 0; i < img_grad_mag.rows; i++)
        for(int j = 0; j < img_grad_mag.cols; j++)
            if(img_grad_mag.at<float>(i, j) > high_threshold)
                img_grad_mag.at<float>(i, j) = 255;
            else if(img_grad_mag.at<float>(i, j)< low_threshold)
                img_grad_mag.at<float>(i, j) = 0;

    for(int i = 0; i < img_grad_mag.rows; i++)
        for(int j = 0; j < img_grad_mag.cols; j++)
            if(img_grad_mag.at<float>(i, j) <= high_threshold && img_grad_mag.at<float>(i, j) >= low_threshold)
            {  
                bool connected = false;
                for(int r = i - neighbour_size; r <= (i + neighbour_size); r++)
                    if(!connected && r >= 0 && r < img_grad_mag.rows)
                        for(int c = j - neighbour_size; c <= (j + neighbour_size); c++)
                            if(c >= 0 && c < img_grad_mag.cols)
                                if(img_grad_mag.at<float>(i, j) > 1)
                                {
                                    connected= true;
                                    break;
                                }
                if(connected)
                    img_grad_mag.at<float>(i, j) = 255;
            }
    return img_grad_mag;
}

Mat corner_detection(Mat img, float threshold, int kernel_w, int kernel_h, int sigma_x, int sigma_y, float alpha, int neighbour_size)
{
    int scale = 1, delta = 0; //constants for Sobel function
    
    Mat img_gray;
    if(img.channels() == 3)
        cvtColor(img, img_gray, CV_BGR2GRAY); //convert input image to gray
    else
        img.copyTo(img_gray);

    Mat img_grad_x, img_grad_y;
    Sobel(img_gray, img_grad_x, CV_32FC1, 1, 0, 3, scale, delta, BORDER_DEFAULT); //apply sobel operator of degree one to direction-x
    Sobel(img_gray, img_grad_y, CV_32FC1, 0, 1, 3, scale, delta, BORDER_DEFAULT); //apply sobel operator of degree one to direction-y
    
    Mat img_grad_x2, img_grad_y2, img_grad_xy;
    img_grad_x2 = img_grad_x.mul(img_grad_x);
    img_grad_y2 = img_grad_y.mul(img_grad_y);
    img_grad_xy = img_grad_x.mul(img_grad_y);
    
    GaussianBlur(img_grad_x2, img_grad_x2, Size(kernel_w, kernel_h), sigma_x, sigma_y); //apply guasian filter
    GaussianBlur(img_grad_y2, img_grad_y2, Size(kernel_w, kernel_h), sigma_x, sigma_y); //apply guasian filter
    GaussianBlur(img_grad_xy, img_grad_xy, Size(kernel_w, kernel_h), sigma_x, sigma_y); //apply guasian filter
    
    Mat img_response = cvCreateMat(img_gray.rows, img_gray.cols, CV_32F);;
    for(int i = 0; i < img_gray.rows; i++)
        for(int j = 0; j < img_gray.cols; j++)
        {
            float arr[2][2] = { {img_grad_x2.at<float>(i, j), img_grad_xy.at<float>(i, j)},
                             {img_grad_xy.at<float>(i, j), img_grad_y2.at<float>(i, j)} };
            Mat M = Mat(2, 2, CV_32FC1, arr);
            Mat eigen_val;
            eigen(M, eigen_val);
            img_response.at<float>(i, j) = eigen_val.at<float>(0) * eigen_val.at<float>(1)
                           - alpha * (eigen_val.at<float>(0) + eigen_val.at<float>(1)) * (eigen_val.at<float>(0) + eigen_val.at<float>(1));
        }
    //img_response = img_grad_x2.mul(img_grad_y2) - img_grad_xy.mul(img_grad_xy) - alpha * (img_grad_x2 + img_grad_y2).mul(img_grad_x2 + img_grad_y2);

    normalize(img_response, img_response, 0, 255, NORM_MINMAX, -1, Mat());
    //img_response.convertTo(img_response, CV_8U);
    //return img_response;
    /*double min, max;
    minMaxLoc(img_response, &min, &max);
    threshold = threshold * max;*/
    for(int i = 0; i < img_response.rows; i++)
    {
        for(int j = 0; j < img_response.cols; j++)
            if(img_response.at<float>(i, j) < threshold)
                img_response.at<float>(i, j) = 0;
            else
            {
                bool local_max = true;
                for(int r = i - neighbour_size; r <= (i + neighbour_size); r++)
                    if(local_max && r >= 0 && r < img_response.rows)
                        for(int c = j - neighbour_size; c <= (j + neighbour_size); c++)
                            if(c >= 0 && c < img_response.cols)
                                if(img_response.at<float>(r, c) > img_response.at<float>(i, j))
                                {
                                    local_max = false;
                                    break;
                                }
                if(local_max)
                    img_response.at<float>(i, j) = 255;
                else
                    img_response.at<float>(i, j) = 0;
            }
    }
    //temp.copyTo(img_response);
    //Mat img_corner;
    img_response.convertTo(img_response, CV_8U);
    for(int i = 0; i < img_response.rows; i++)
        for(int j = 0; j < img_response.cols; j++)
            if(img_response.at<float>(i, j) > 0)
                circle( img_gray, Point( i, j ), 5,  Scalar(255), 2, 8, 0 );
    //return img_response;
    return img_gray;
}

Mat hough_line_detection(Mat img, float threshold, int neighbour_size, int theta_step, int p_step)
{
    int scale = 1, delta = 0; //constants for Sobel function

    Mat img_edge, img_grad_x, img_grad_y, img_gray;
    
    Canny(img, img_edge, 75, 200, 3, true);

    if(img.channels() == 3)
        cvtColor(img, img_gray, CV_BGR2GRAY); //convert input image to gray
    else
        img.copyTo(img_gray);

    Sobel(img_gray, img_grad_x, CV_32FC1, 1, 0, 3, scale, delta, BORDER_DEFAULT); //apply sobel operator of degree one to direction-x
    Sobel(img_gray, img_grad_y, CV_32FC1, 0, 1, 3, scale, delta, BORDER_DEFAULT); //apply sobel operator of degree one to direction-y
    
    Mat img_grad_theta = Mat::zeros(img_edge.rows, img_edge.cols, CV_32F);
    
    for(int y = 0; y < img_grad_theta.rows; y++)
        for(int x = 0; x < img_grad_theta.cols; x++)
        {
            float angle = atan2(img_grad_y.at<float>(y, x), img_grad_x.at<float>(y, x)) * 180 / CV_PI;
            if(angle < 0.0)
                angle += 180.0;
            img_grad_theta.at<float>(y, x) = angle;
        }
    
    int n = img.rows, m = img.cols, max_p = max(n, m), lag = 2;
    Mat hough_line_space = Mat::zeros(180, 2 * max_p + lag, CV_32F);
    
    for(int y = 0; y < img_edge.rows; y++)
        for(int x = 0; x < img_edge.cols; x++)
        {
            if(img_edge.at<uint8_t>(y, x) < 255)
                continue;
            int theta = approximate(img_grad_theta.at<float>(y, x), theta_step);
            int p = approximate(y * sin((double)theta * CV_PI / 180) + x * cos((double)theta * CV_PI / 180) + max_p, p_step);
            hough_line_space.at<float>(theta, p) += 1.0;
        }
    
    Mat img_line = Mat::zeros(img.rows, img.cols, CV_8U);
    float max_v = 0;
    
    for(int theta = 0; theta < hough_line_space.rows; theta += theta_step)
        for(int p = 0; p < hough_line_space.cols; p += p_step)
        {
            //cout<<theta<<" "<<p<<"\n";
            max_v = max(max_v, hough_line_space.at<float>(theta, p));
            if(hough_line_space.at<float>(theta, p) > threshold)
            {
                bool local_max = true;
                for(int i = theta - neighbour_size * theta_step; i <= theta + neighbour_size * theta_step; i += theta_step)
                    if(local_max && i >= 0 && i < hough_line_space.rows)
                        for(int j = p - neighbour_size * p_step; j <= p + neighbour_size * p_step; j += p_step)
                            if(j >= 0 && j < hough_line_space.cols)
                                if(hough_line_space.at<float>(theta, p) < hough_line_space.at<float>(i, j))
                                {
                                    local_max = false;
                                    break;
                                }
                if(local_max)
                    draw_line(img_line, theta, p - max_p);
            }
        }
    return img_line;
}

void draw_line(Mat & img, int theta, int p)
{
    if(theta > 0)
    {
        for(int x = 0; x < img.cols; x++)
        {
            int y = (p - x * cos((double)theta * CV_PI / 180)) / sin((double)theta * CV_PI / 180);
            if(y >= 0 && y < img.rows)
                img.at<uint8_t>(y, x) = 255;
        }
    }
    else
    {
        for(int y = 0; y < img.rows; y++)
        {
            int x = (p - y * sin((double)theta * CV_PI / 180)) / cos((double)theta * CV_PI / 180);
            if(x >= 0 && x < img.cols)
                img.at<uint8_t>(y, x) = 255;
        }    
    }
    cout<<"y\n";
}

int approximate(float x, int step)
{
    int temp = round(x/step);
    return temp*step;
}

Mat hough_circle_detection(Mat img, float threshold, int neighbour_size, int rad_step, int center_step)
{
    Mat img_edge, img_circle;
    Canny(img, img_edge, 75, 200, 3, true);
    //return img_edge;
    int max_r = min(img_edge.rows, img_edge.cols) / 2;
    img_circle = Mat::zeros(img.rows, img.cols, CV_8U);
    
    for(int r = rad_step; r < max_r; r += rad_step)
    {
        Mat circle_centers = Mat::zeros(img.rows, img.cols, CV_32F);
        for(int y = 0; y < img_edge.rows; y++)
            for(int x = 0; x < img_edge.cols; x++)
            {
                if(img_edge.at<uint8_t>(y, x) < 255)
                    continue;
                for(int a = rad_step; a < img_edge.rows - rad_step; a += center_step)
                {
                    if( r * r - (y - a) * (y - a) < 0.0)
                        continue;
                    int b1 = approximate(x - sqrt(r * r - (y - a) * (y - a)), center_step);
                    int b2 = approximate(x + sqrt(r * r - (y - a) * (y - a)), center_step);
                    if(b1 >= rad_step && b1 < img_edge.cols - rad_step)
                        circle_centers.at<float>(a, b1) += 1.0;
                    if(b2 >= rad_step && b2 < img_edge.cols - rad_step)
                        circle_centers.at<float>(a, b2) += 1.0;                    
                }
            }
        for(int a = rad_step; a < img_edge.rows - rad_step; a += center_step)
            for(int b = rad_step; b < img_edge.cols - rad_step; b += center_step)
            {
                if(circle_centers.at<float>(a, b) > threshold)
                {
                    bool local_max = true;
                    for(int i = a - neighbour_size * center_step; i <= a + neighbour_size * center_step; i += center_step)
                        if(local_max && i >= rad_step && i < img_edge.rows - rad_step)
                            for(int j = b - neighbour_size * center_step; j <= b + neighbour_size * center_step; j += center_step)
                                if(j >= rad_step && j < img_edge.cols - rad_step)
                                    if(circle_centers.at<float>(a, b) < circle_centers.at<float>(i, j))
                                    {
                                        local_max = false;
                                        break;
                                    }
                    
                    if(local_max)
                    {
                        cout<<circle_centers.at<float>(a, b)<<" ";
                        draw_circle(img_circle, a, b, r);
                    }
                }
            }
    }
    return img_circle;
}

void draw_circle(Mat & img, int a, int b, int r)
{
    circle(img, Point(b, a), r, Scalar(255));
    cout<<a<<" "<<b<<" "<<r<<endl;
}

Mat derivative_x(Mat img1, Mat img2)
{
    float arr[2][2] = { {-1.0, 1.0}, {-1.0, 1.0} };
    Mat filter = Mat(2, 2, CV_32FC1, arr), img1_x, img2_x;
    filter2D(img1, img1_x, CV_32FC1, filter);
    filter2D(img2, img2_x, CV_32FC1, filter);
    return img1_x + img2_x;
}

Mat derivative_y(Mat img1, Mat img2)
{
    float arr[2][2] = { {-1.0, -1.0}, {1.0, 1.0} };
    Mat filter = Mat(2, 2, CV_32FC1, arr), img1_y, img2_y;
    filter2D(img1, img1_y, CV_32FC1, filter);
    filter2D(img2, img2_y, CV_32FC1, filter);
    return img1_y + img2_y;
}

Mat color_coding(Mat x, Mat y)
{
    Mat magnitude, angle;
    cartToPolar(x, y, magnitude, angle, true);

    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    merge(_hsv, 3, hsv);

    //convert to BGR and show
    Mat bgr;//CV_32FC3 matrix
    cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}


Mat derivative_t(Mat img1, Mat img2)
{
    float arr[2][2] = { {-1.0, -1.0}, {-1.0, -1.0} };
    Mat filter = Mat(2, 2, CV_32FC1, arr), img1_t, img2_t;
    filter2D(img1, img1_t, CV_32FC1, filter);
    filter *= -1;
    filter2D(img2, img2_t, CV_32FC1, filter);
    return img1_t + img2_t;
}

Mat optical_flow_lucas_kanade(Mat img1, Mat img2, int w)
{
    Mat img_x = derivative_x(img1, img2), img_y = derivative_y(img1, img2), img_t = derivative_t(img1, img2);
    Mat u = Mat::zeros(img1.rows, img1.cols, CV_32F), v = Mat::zeros(img1.rows, img1.cols, CV_32F);
    for(int i = 0; i < img1.rows; i++)
        for(int j = 0; j < img1.cols; j++)
        {
            float sum_f_x_2 = 0, sum_f_y_2 = 0, sum_f_x_y = 0, sum_f_x_t = 0, sum_f_y_t = 0;
            for(int l = max(0, i - w); l <= min(i + w, img1.rows-1); l++)
                for(int r = max(0, j - w); r <= min(j + w, img1.cols-1); r++)
                {
                    sum_f_x_2 += img_x.at<float>(l, r) * img_x.at<float>(l, r);
                    sum_f_y_2 += img_y.at<float>(l, r) * img_y.at<float>(l, r);
                    sum_f_x_y += img_x.at<float>(l, r) * img_y.at<float>(l, r);
                    sum_f_x_t += img_x.at<float>(l, r) * img_t.at<float>(l, r);
                    sum_f_y_t += img_y.at<float>(l, r) * img_t.at<float>(l, r);
                }
            u.at<float>(i, j) = ((-sum_f_y_2 * sum_f_x_t) + (sum_f_x_y * sum_f_y_t)) / ((sum_f_x_2 * sum_f_y_2) - (sum_f_x_y * sum_f_x_y));
            v.at<float>(i, j) = ((sum_f_x_t * sum_f_x_y) - (sum_f_x_2 * sum_f_y_t)) / ((sum_f_x_2 * sum_f_y_2) - (sum_f_x_y * sum_f_x_y));
            float angle = (atan2(v.at<float>(i, j), u.at<float>(i, j)) * 180 / CV_PI) + 180;
        }
    return color_coding(u, v);
}

Mat optical_flow_horn_schunk(Mat img1, Mat img2, float lambda, int iter)
{
    Mat img_x = derivative_x(img1, img2), img_y = derivative_y(img1, img2), img_t = derivative_t(img1, img2);
    Mat u_current = Mat::zeros(img1.rows, img1.cols, CV_32F), v_current = Mat::zeros(img1.rows, img1.cols, CV_32F);
    Mat u_last, v_last;
    for(int k = 0; k < iter; k++)
    {
        u_current.copyTo(u_last);
        v_current.copyTo(v_last);
        for(int i = 0; i < img1.rows; i++)
            for(int j = 0; j < img1.cols; j++)
            {
                float u_avg = 0.0, v_avg = 0.0, n = 0;
                if(i-1 >= 0)
                {
                    u_avg += u_last.at<float>(i-1, j);
                    v_avg += v_last.at<float>(i-1, j);
                    n++;
                }
                if(i+1 < img1.cols)
                {
                    u_avg += u_last.at<float>(i+1, j);
                    v_avg += v_last.at<float>(i+1, j);
                    n++;
                }
                if(j-1 >= 0)
                {
                    u_avg += u_last.at<float>(i, j-1);
                    v_avg += v_last.at<float>(i, j-1);
                    n++;
                }
                if(j+1 < img1.rows)
                {
                    u_avg += u_last.at<float>(i, j+1);
                    v_avg += v_last.at<float>(i, j+1);
                    n++;
                }
                u_avg /= n;
                v_avg /= n;

                float p = img_x.at<float>(i, j) * u_avg + img_y.at<float>(i, j) * v_avg + img_t.at<float>(i, j);
                float d = lambda + img_x.at<float>(i, j) * img_x.at<float>(i, j) + img_y.at<float>(i, j) * img_y.at<float>(i, j);

                u_current.at<float>(i, j) = u_avg - img_x.at<float>(i, j) * p / d;
                v_current.at<float>(i, j) = v_avg - img_y.at<float>(i, j) * p / d;
            }
    }
    return color_coding(u_current, v_current);
}
