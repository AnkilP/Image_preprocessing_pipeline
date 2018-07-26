#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <iostream>

//#include <Eigen/Core>

//static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter
{
  unsigned char lut[256];
    float fGamma = 1.5;
    cv::Mat str_img;
  cv::Mat cpy;

  cv::Mat grad_x_stream;
  cv::Mat grad_y_stream;

    int lowThreshold = 100;
    int ratio = 5;
    int kernel_size = 7;

    float color_correction_matrix[9] = {0.61, 0.13, -0.02, -0.43, 1.02, 0.42, 0.04, -0.15, 1.2};
    cv::Mat color_corr_matrix = cv::Mat(3, 4, CV_64F, color_correction_matrix);

public:
    ImageConverter(const cv::Mat &img) {
        str_img = img;
        cpy = str_img.clone();
        std::cout << "Created image preprocessing pipeline" << std::endl;
  }
  ~ImageConverter()
  {
    std::cout << "/* Destroyed ImageConverter */" << '\n';
  }

    void dead_pixel_concealment();//const cv::Mat& str_img, cv::Mat&cpy);
    void black_level_correction(double contrast, double brightness);

    void GammaCorrection();

    void lut_builder();
  float rgb2luma(float r, float g, float b);

    void bilateral_filter();

    void anti_aliasing(const cv::Mat &str_img, cv::Mat cpy);

    void histogram_equalization(int clip);

    void chroma_noise_filter();

    void hue_saturation_control(const cv::Mat &str_img, cv::Mat &cpy);

    void automatic_white_balance_gain_control(const cv::Mat &str_img, cv::Mat &cpy);

    void display_image();

    void edge_finder();

    void edge_enhancement();
};

void ImageConverter::lut_builder() {
    for (int i = 0; i < 256; i++)
	{
		lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}
}

void ImageConverter::display_image() {
    //cv::Mat display_temp;
    //cv::hconcat(str_img,cpy,display_temp);
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("gotstreets?", str_img);
    cv::waitKey(0); // Wait for a keystroke in the window
    cv::imshow("gotstreets?", cpy);
    cv::waitKey(0);
}

// don't forget to include related head files - not sure if the opengl route might be easier given this problem's ubiquity in this space
//https://gist.github.com/zhangzhensong/03f67947c22acb5ee922
/*void BindCVMat2GLTexture(cv::Mat& image, GLuint& imageTexture)
{
   if(image.empty()){
      std::cout << "image empty" << std::endl;
  }else{
      //glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      glGenTextures(1, &imageTexture1);
      glBindTexture(GL_TEXTURE_2D, imageTexture1);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Set texture clamping method
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

      cv::cvtColor(image, image, CV_RGB2BGR);

      glTexImage2D(GL_TEXTURE_2D,         // Type of texture
                     	0,                   // Pyramid level (for mip-mapping) - 0 is the top level
			GL_RGB,              // Internal colour format to convert to
                     	image.cols,          // Image width  i.e. 640 for Kinect in standard mode
                     	image.rows,          // Image height i.e. 480 for Kinect in standard mode
                     	0,                   // Border width in pixels (can either be 1 or 0)
			GL_RGB,              // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
			GL_UNSIGNED_BYTE,    // Image data type
			image.ptr());        // The actual image data itself
	}
} */

//dead pixel concealment using median filter
void ImageConverter::dead_pixel_concealment() {
    cv::medianBlur(str_img, cpy, 1); //1 is the kernel size and ideal for concealing dead pixels
}

//gamma level compensation: https://github.com/DynamsoftRD/opencv-programming/blob/master/gamma-correction/gamma.cpp
void ImageConverter::GammaCorrection() {
    //cv::CV_Assert(str_img.data);
    // accept only char type matrices
    //cv::CV_Assert(str_img.depth() != sizeof(uchar));
    assert(lut != NULL);
    const int channels = cpy.channels();
    switch (channels)
    {
        case 1:
        {
            cv::MatIterator_<uchar> it, end;
            for (it = cpy.begin<uchar>(), end = cpy.end<uchar>(); it != end; it++)
                //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
                *it = lut[(*it)];
            break;
        }
        case 3:
        {
            cv::MatIterator_<cv::Vec3b> it, end;
            for (it = cpy.begin<cv::Vec3b>(), end = cpy.end<cv::Vec3b>(); it != end; it++)
            {
                (*it)[0] = lut[((*it)[0])];
                (*it)[1] = lut[((*it)[1])];
                (*it)[2] = lut[((*it)[2])];
            }

            break;
        }
    }
}

//lens shading correction:
//void ImageConverter::lens_shading_correction(){
//
//}
////////////////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
//anti-aliasing filter - MLAA: http://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/reshetov09_mlaa.pdf
float ImageConverter::rgb2luma(float r, float g, float b){
    return sqrt(r*0.299 + 0.587*g + 0.114*b);
}

void ImageConverter::bilateral_filter() {
    cv::bilateralFilter(cpy, cpy, 1, 2, 0.5);
}

//void ImageConverter::edge_detector(const cv::Mat & str_img, cv::Mat & cpy) {
//    cv::Canny(&str_img, &cpy, lowThreshold, lowThreshold * ratio, kernel_size);
//}

//MLAA
/*
1.Find discontinuities between pixels in a given image.
2.Identify U-shaped, Z-shaped, L-shaped patterns.
3.Blend colors in the neighborhood of these patterns.
*/

//helper functions
//bool ImageConverter::determine_vertical_surroundings(const cv::Mat & str_img, int index_i, int index_j){
//    if(abs(str_img.at<double>(i,j) - str_img.at<double>(i+1,j)) >= )
//}

/*
 * At the end of the first step, each pixel is marked with the horizontal discontinuity flag
 * and/or the vertical discontinuity flag, if such discontinuities are detected.
 */
void ImageConverter::edge_finder() {
//    std::vector<cv::Mat> channels[3];
//    cv::split(str_img, channels);
//    cv::Mat hue_channel = channels[0]; //L in lut or H in HSV
    /*for(int i = 1; i < size(str_img.rows) - 1; ++i){
	    for(int j = 1; j < size(str_img.cols) - 1; ++j){
            if(str_img.at<double>(i+1,j)){
                cpy.at<double>(i+1, j)++;
            }
            if(str_img.at<double>(i, j+1) =  )
	    }
	}*/
    //assuming this is in LUT space
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;
    cv::Canny(str_img, cpy, lowThreshold, lowThreshold * ratio, kernel_size);
}

//void ImageConverter::find_Z_pattern(cv::Mat & binary_mask){
//
//    if(grad_x_stream.at<double>())
//}
//
//void ImageConverter::anti_aliasing(const cv::Mat & str_img, cv::Mat cpy) {
//    //combine all previous operations
//}
//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\////////////////////////////////////////////////

void ImageConverter::histogram_equalization(int clip) {
    //converting image to LAB space
    cv::Mat lab_image(str_img.rows, str_img.cols, str_img.type());
    cv::cvtColor(str_img, lab_image, CV_BGR2Lab);
    //extract L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]
    //applying CLAHE on L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    //cv::Mat cpy;
    clahe->apply(lab_planes[0], cpy);
    // Merge the the color planes back into an Lab image
    cpy.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);
    // convert back to RGB
    cv::cvtColor(lab_image, cpy, CV_Lab2BGR);
}

//awb gain control // after gamma correction
/*
https://pdfs.semanticscholar.org/a681/674e7657a5b7f02ac78cd274d0d1a54072b6.pdf
*/
//void ImageConverter::automatic_white_balance_gain_control(const cv::Mat & str_img, cv::Mat & cpy) {
//
//}

//cfa interpolation
/*
Image sensors have pixels that are dedicated to a single colour and thus each pixel must be interpolated
Only use this if you have access to the GPU!!
This is not really useful for us since the camera handles this anyway
*/
//
//void ImageConverter::demosaicing(const cv::Mat & str_img, cv::Mat & cpy) {
//    //TODO: add check for nvcc
//    try {
//        cv::cuda::demosaicing(str_img, cpy, cv::COLOR_BayerBG2BGR_MHT, 3);
//    }
//    catch (const cv::Exception& ex) {
//        std::_Count_pr << "Error: " << ex.what() << std::endl;
//    }
//}

//black level correction
void ImageConverter::black_level_correction(double contrast, double brightness) {
    //linear transform method
    for (int y = 0; y < str_img.rows; y++) {
        for (int x = 0; x < str_img.cols; x++) {
            for( int c = 0; c < 3; c++ ) {
                cpy.at<cv::Vec3b>(y, x)[c] =
                        cv::saturate_cast<uchar>(contrast * (str_img.at<cv::Vec3b>(y, x)[c]) + brightness);
            }
        }
    }
}
//color correction
/*
RGB_out = (alpha * A I_w * RGB_in)^gamma
where alpha, I_W and A, respectively represent the exposure compensation
gain, the diagonal matrix for the illuminant compensation
and the color matrix transformation
*/
//void ImageConverter::color_correction(const cv::Mat & str_img, cv::Mat & cpy) {
//    int i,j;
//    uchar* p;
//    for( i = 0; i < nRows; ++i)
//    {
//        p = I.ptr<uchar>(i);
//        for ( j = 0; j < nCols; ++j)
//        {
//            p[j] = table[p[j]];
//        }
//    }
//    return I;
//}

//color space conversion
/*
Moving to another color space (with luminance) will allow me to use more
conventional grayscale algorithms for image processing while still being able to move back to the
BGR space easily
*/
//void ImageConverter::color_space_conversion_out_of_BGR(const cv::Mat & str_img, cv::Mat & cpy) {
//    //can run into issues where the input file is not BGR but that should be apparent from
//    //compilation
//    try {
//        cv::cvtColor(str_img, cpy, cv::COLOR_BGR2Lab);
//    }
//    catch (cv::Exception & e) {
//        std::cout << e.what() << std::endl;
//    }
//
//}
//
//void ImageConverter::color_space_conversion_into_BGR(const cv::Mat & str_img, cv::Mat & cpy) {
//    //can run into issues where the input file is not Lab but that should be apparent from
//    //compilation
//    try {
//        cv::cvtColor(str_img, cpy, cv::COLOR_Lab2BGR);
//    }
//    catch (cv::Exception & e) {
//        std::cout << e.what() << std::endl;
//    }
//
//}
//noise filter for chroma
//http://inf.ufrgs.br/~eslgastal/AdaptiveManifolds/
/*
 *     Parameters:
        src – Input 8-bit 3-channel image.
        dst – Output image with the same size and type as src .
        templateWindowSize – Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels
        searchWindowSize – Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels
        h – Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise
        hColor – The same as h but for color components. For most images value equals 10 will be enought to remove colored noise and do not distort colors
 */
//void ImageConverter::chroma_noise_filter() {
//    cv::fastNlMeansDenoisingColored(str_img, cpy, 3, 3, 7, 21);
//
//}
//
////hue saturation control
//void ImageConverter::hue_saturation_control(const cv::Mat & str_img, cv::Mat cpy) {
//
//}
//
////noise filter for luma
//
////edge enhancement
//void ImageConverter::edge_enhancement() {
//    int kernel = 3;
//    cv::filter2D(str_img, cpy, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
//}

//contrast brightness control

//data formatter


