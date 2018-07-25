//
// Created by ankilp on 25/07/18.
//

#include <iostream>

#include "testing_environment.hpp"
#include "image_processing.hpp"

int main(int argc, char **argv) {
    cv::Mat street_image = cv::imread("street.jpg");
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
    if (street_image.empty())                      // Check for invalid input
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cv::imshow("Display window", street_image);                // Show our image inside it.
    cv::waitKey(5000); // Wait for a keystroke in the window
    ImageConverter imageConverter();
    //first let's convert the image to Lab or YCrCb format
    return 0;
}