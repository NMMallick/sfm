#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

// #ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
    "{ help h | | Print help Message. }"
    "{ @input1 | | Path to input 1. }"
    "{ @input2 | | Path to input 2. }";

int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );

    Mat img1 = imread( samples::findFile( parser.get<String>( "@input1" ) ), IMREAD_GRAYSCALE );
    Mat img2 = imread( samples::findFile( parser.get<String>( "@input2" ) ), IMREAD_GRAYSCALE );

    if (img1.empty() || img2.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    //-- Step 1: Detect the keypoints using SIFT Detector
    int minHessian = 500;
    Ptr<SIFT> detector = SIFT::create( minHessian );
    std::vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;

    detector->detectAndCompute(img1, noArray(), kp1, desc1);
    detector->detectAndCompute(img2, noArray(), kp2, desc2);


    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(desc1, desc2, knn_matches, 2);

    //-- Filter matches uing the Lowe's ratio test
    const float ratio_thresh = 110.0f;
    std::vector<DMatch> good_matches;

    for (size_t i = 0; i< knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    if( good_matches.size() == 0)
        std::cout << "No good matches" << std::endl;

    //-- Draw keypoints
    Mat img_matches;
    drawMatches( img1, kp1, img2, kp2, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Resize images
    int down_width = 1400;
    int down_height = 800;
    Mat resized_down;

    resize(img_matches, resized_down, Size(down_width, down_height), INTER_LINEAR);

    //-- Show detected (drawn) keypoints
    imshow("SIFT Matches", resized_down);
    waitKey();
    return 0;
}
// #else
// int main()
// {
//     std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
//     return 0;
// }
// #endif