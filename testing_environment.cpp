//
// Created by ankilp on 25/07/18.
//

#include <iostream>

#include "testing_environment.hpp"
#include "adaptive_manifold.hpp"

#include "gtest/gtest.h"

TEST (Log2, TrialTest){
EXPECT_EQ (0, Image_Preprocessing::Log2(1.0));
EXPECT_EQ (-1, Image_Preprocessing::Log2(-1.0));
}

TEST (differences, DIFFXY){
    EXPECT_EQ(cv::Zeros(cv::Scalar(1)), Image_Preprocessing::diffX(src, dst));
    EXPECT_EQ(cv::Zeros(cv::Scalar(1)), Image_Preprocessing::diffY(src, dst));
}

class Test_Adaptive_Manifold : public cvtest::BaseTest{
private:
    RUN_ALL_TESTS();
public:
    cv::Mat_<cv::Point3f> src;
    cv::Mat_<cv::Point3f> dst;
};


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    cv::Mat street_image = cv::imread("street.jpg");
    if (street_image.empty())                      // Check for invalid input
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    //cv::imshow("Display window", street_image);  // Show our image inside it.
    ImageConverter imageConverter(street_image);
    imageConverter.edge_finder();
    imageConverter.display_image();
    //first let's convert the image to Lab or YCrCb format
    return 0;
}