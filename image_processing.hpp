#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <iostream>

//#include <Eigen/Core>

//static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter
{
  unsigned char lut[256];
  float fGamma = 0.5;
  cv::Mat img;
  cv::Mat cpy;

  cv::Mat grad_x_stream;
  cv::Mat grad_y_stream;

  int lowThreshold = 1;
  int ratio = 3;
  int kernel_size = 3;

    //
    float color_correction_matrix[9] = {0.61, 0.13, -0.02, -0.43, 1.02, 0.42, 0.04, -0.15, 1.2};
    cv::Mat color_corr_matrix = cv::Mat(3, 4, CV_64F, color_correction_matrix);

public:
    ImageConverter(const cv::Mat &img) {
	  std::cout << "Not using ros" << std::endl;
  }
  ~ImageConverter()
  {
    std::cout << "/* Destroyed ImageConverter */" << '\n';
  }
  void dead_pixel_concealment(const cv::Mat& src, cv::Mat&dst);
  void black_level_correction(const cv::Mat &src, cv::Mat &dst, const double & contrast, const int & brightness);
  void GammaCorrection(const cv::Mat& src, cv::Mat& dst, float & fGamma);
  void lut_builder(float fGamma);
  float rgb2luma(float r, float g, float b);
  void bilateral_filter(cv::Mat & src, cv::Mat & dst);
  void anti_aliasing(const cv::Mat & src, cv::Mat dst);
    
  void histogram_equalization(const cv::Mat & src, cv::Mat & image_clahe, const int clip);
  void chroma_noise_filter(const cv::Mat & src, cv::Mat & dst);
  void hue_saturation_control(const cv::Mat & src, cv::Mat & dst);
  void automatic_white_balance_gain_control(const cv::Mat & src, cv::Mat & dst);

    void edge_detector(const cv::Mat &src);
};

void ImageConverter::lut_builder(float fGamma){
    for (int i = 0; i < 256; i++)
	{
		lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}
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
void ImageConverter::dead_pixel_concealment(const cv::Mat& src, cv::Mat& dst){
    cv::medianBlur(src, dst, 1); //1 is the kernel size
}

//gamma level compensation: https://github.com/DynamsoftRD/opencv-programming/blob/master/gamma-correction/gamma.cpp
void ImageConverter::GammaCorrection(const cv::Mat& src, cv::Mat& dst, float & fGamma){
    //cv::CV_Assert(src.data);
    // accept only char type matrices
    //cv::CV_Assert(src.depth() != sizeof(uchar));
    const int channels = dst.channels();
    switch (channels)
    {
        case 1:
        {
            cv::MatIterator_<uchar> it, end;
            for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
                //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
                *it = lut[(*it)];
            break;
        }
        case 3:
        {
            cv::MatIterator_<cv::Vec3b> it, end;
            for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
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

void ImageConverter::bilateral_filter(cv::Mat & src, cv::Mat & dst){
    cv::bilateralFilter(src, dst, 1, 2, 0.5);
}

//void ImageConverter::edge_detector(const cv::Mat & src, cv::Mat & dst) {
//    cv::Canny(&src, &dst, lowThreshold, lowThreshold * ratio, kernel_size);
//}

//MLAA
/*
1.Find discontinuities between pixels in a given image.
2.Identify U-shaped, Z-shaped, L-shaped patterns.
3.Blend colors in the neighborhood of these patterns.
*/

//helper functions
//bool ImageConverter::determine_vertical_surroundings(const cv::Mat & src, int index_i, int index_j){
//    if(abs(src.at<double>(i,j) - src.at<double>(i+1,j)) >= )
//}

/*
 * At the end of the first step, each pixel is marked with the horizontal discontinuity flag
 * and/or the vertical discontinuity flag, if such discontinuities are detected.
 */
//void ImageConverter::edge_finder(const cv::Mat & src) {
//    std::vector<cv::Mat> channels[3];
//    cv::split(src, channels);
//    cv::Mat hue_channel = channels[0]; //L in lut or H in HSV
//    /*for(int i = 1; i < size(src.rows) - 1; ++i){
//	    for(int j = 1; j < size(src.cols) - 1; ++j){
//            if(src.at<double>(i+1,j)){
//                dst.at<double>(i+1, j)++;
//            }
//            if(src.at<double>(i, j+1) =  )
//	    }
//	}*/
//    //assuming this is in LUT space
//    int ddepth = cv::CV_16S;
//    int scale = 1;
//    int delta = 0;
//
//    cv::Scharr( hue_channel, grad_x_stream, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
//    cv::Scharr( hue_channel, grad_y_stream, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
//}
//
//void ImageConverter::find_Z_pattern(cv::Mat & binary_mask){
//
//    if(grad_x_stream.at<double>())
//}

void ImageConverter::anti_aliasing(const cv::Mat & src, cv::Mat dst) {
    //combine all previous operations
}
//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\////////////////////////////////////////////////

void ImageConverter::histogram_equalization(const cv::Mat & src, cv::Mat & image_clahe, const int clip){
    //converting image to LAB space
    cv::Mat lab_image;//(src.rows, src.cols, original_image.type());
    cv::cvtColor(src, lab_image, CV_BGR2Lab);
    //extract L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]
    //applying CLAHE on L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);
    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);
    // convert back to RGB
    cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
}

//awb gain control // after gamma correction
/*
https://pdfs.semanticscholar.org/a681/674e7657a5b7f02ac78cd274d0d1a54072b6.pdf
*/
//void ImageConverter::automatic_white_balance_gain_control(const cv::Mat & src, cv::Mat & dst) {
//
//}

//cfa interpolation
/*
Image sensors have pixels that are dedicated to a single colour and thus each pixel must be interpolated
Only use this if you have access to the GPU!!
This is not really useful for us since the camera handles this anyway
*/
//
//void ImageConverter::demosaicing(const cv::Mat & src, cv::Mat & dst) {
//    //TODO: add check for nvcc
//    try {
//        cv::cuda::demosaicing(src, dst, cv::COLOR_BayerBG2BGR_MHT, 3);
//    }
//    catch (const cv::Exception& ex) {
//        std::_Count_pr << "Error: " << ex.what() << std::endl;
//    }
//}

//black level correction
void ImageConverter::black_level_correction(const cv::Mat & src, cv::Mat & dst, const double & contrast, const int & brightness){
    //linear transform method
    for( int y = 0; y < src.rows; y++ ) {
        for( int x = 0; x < src.cols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                dst.at<cv::Vec3b>(y,x)[c] =
                        cv::saturate_cast<uchar>( contrast*( src.at<cv::Vec3b>(y,x)[c] ) + brightness );
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
//void ImageConverter::color_correction(const cv::Mat & src, cv::Mat & dst) {
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
//void ImageConverter::color_space_conversion_out_of_BGR(const cv::Mat & src, cv::Mat & dst) {
//    //can run into issues where the input file is not BGR but that should be apparent from
//    //compilation
//    try {
//        cv::cvtColor(src, dst, cv::COLOR_BGR2Lab);
//    }
//    catch (cv::Exception & e) {
//        std::cout << e.what() << std::endl;
//    }
//
//}
//
//void ImageConverter::color_space_conversion_into_BGR(const cv::Mat & src, cv::Mat & dst) {
//    //can run into issues where the input file is not Lab but that should be apparent from
//    //compilation
//    try {
//        cv::cvtColor(src, dst, cv::COLOR_Lab2BGR);
//    }
//    catch (cv::Exception & e) {
//        std::cout << e.what() << std::endl;
//    }
//
//}
//noise filter for chroma
//void ImageConverter::chroma_noise_filter(const cv::Mat & src, cv::Mat & dst) {
//    //cv::fastNlMeansDenoisingColored(src, dst, 3, 3, 7, 21);
//
//}
//
////hue saturation control
//void ImageConverter::hue_saturation_control(const cv::Mat & src, cv::Mat dst) {
//
//}
//
////noise filter for luma
//
////edge enhancement
//void ImageConverter::edge_enhancement(const cv::Mat & src, cv::Mat dst) {
//    //kernel = ;
//    //cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
//}

//contrast brightness control

//data formatter


