//
// Created by some1 on 8/2/2018.
//

#ifndef IMAGE_PREPROCESSING_PIPELINE_ADAPTIVE_MANIFOLD_HPP
#define IMAGE_PREPROCESSING_PIPELINE_ADAPTIVE_MANIFOLD_HPP


#include "adaptive_manifold.hpp"
#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>

class Image_Preprocessing{
protected: //TODO: allocate memory for two images
    cv::Mat image;
    cv::Mat_ cpy_image;
public:
    Image_Preprocessing();
    inline double Log2(double n)
    {
        return log(n) / log(2.0);
    }

    inline int computeManifoldTreeHeight(double sigma_s, double sigma_r)
    {
        const double Hs = floor(Log2(sigma_s)) - 1.0;
        const double Lr = 1.0 - sigma_r;
        return std::max(2, static_cast<int>(ceil(Hs * Lr)));
    }

    inline double floor_to_power_of_two(double r)
    {
        return pow(2.0, floor(Log2(r)));
    }

    void channelsSum(const Mat_<cv::Point3f>& src, cv::Mat_<float>& dst);
    void phi(const cv::Mat_<float>& src, cv::Mat_<float>& dst, float sigma);
    void catCn(const cv::Mat_<cv::Point3f>& a, const cv::Mat_<float>& b, cv::Mat_<cv::Vec4f>& dst);

    template <typename T>
    virtual void ensureSizeIsEnough(int rows, int cols, cv::Mat_<T>& m);

    template <typename T>
    virtual void ensureSizeIsEnough(const cv::Size, cv::Mat_<T>& m);

    template <typename T>
    virtual void h_filter(const cv::Mat_<T>& src, cv::Mat_<T>& dst, float sigma);

    template <typename T>
    virtual void rdivide(const cv::Mat_<T>& a, const cv::Mat_<float>& b, cv::Mat_<T>& dst);

    template <typename T>
    virtual void times(const cv::Mat_<T>& a, const cv::Mat_<float>& b, cv::Mat_<T>& dst)
    //virtual void apply_algorithm();
    //virtual ~Image_Preprocessing();
};

Image_Preprocessing::Image_Preprocessing() {
    image = cv::Zeros(cv::Size(1920,1200)); //TODO: Rosify to get input from camera OR camera input from our own camera driver
    cpy_image = image.clone();
}

class Adaptive_Manifold : Image_Preprocessing {
private:
    cv::cv::Mat_<cv::Point3f> eta_1;
    cv::Mat_<cv::uchar> cluster_1;

    cv::Mat_<cv::Point3f> tilde_dst;
    cv::Mat_<float> alpha;
    cv::Mat_<cv::Point3f> diff;
    cv::Mat_<cv::Point3f> dst;

    cv::Mat_<float> V;

    cv::Mat_<cv::Point3f> dIcdx;
    cv::Mat_<cv::Point3f> dIcdy;
    cv::Mat_<float> dIdx;
    cv::Mat_<float> dIdy;
    cv::Mat_<float> dHdx;
    cv::Mat_<float> dVdy;

    cv::Mat_<float> t;

    cv::Mat_<float> theta_masked;
    cv::Mat_<cv::Point3f> mul;
    cv::Mat_<cv::Point3f> numerator;
    cv::Mat_<float> denominator;
    cv::Mat_<cv::Point3f> numerator_filtered;
    cv::Mat_<float> denominator_filtered;

    cv::Mat_<cv::Point3f> X;
    cv::Mat_<cv::Point3f> eta_k_small;
    cv::Mat_<cv::Point3f> eta_k_big;
    cv::Mat_<cv::Point3f> X_squared;
    cv::Mat_<float> pixel_dist_to_manifold_squared;
    cv::Mat_<float> gaussian_distance_weights;
    cv::Mat_<cv::Point3f> Psi_splat;
    cv::Mat_<Vec4f> Psi_splat_joined;
    cv::Mat_<Vec4f> Psi_splat_joined_resized;
    cv::Mat_<Vec4f> blurred_projected_values;
    cv::Mat_<cv::Point3f> w_ki_Psi_blur;
    cv::Mat_<float> w_ki_Psi_blur_0;
    cv::Mat_<cv::Point3f> w_ki_Psi_blur_resized;
    cv::Mat_<float> w_ki_Psi_blur_0_resized;
    cv::Mat_<float> rand_vec;
    cv::Mat_<float> v1;
    cv::Mat_<float> Nx_v1_mult;
    cv::Mat_<float> theta;

    std::vector<cv::Mat_<cv::Point3f>> eta_minus;
    std::vector<cv::Mat_<cv::uchar>> cluster_minus;
    std::vector<cv::Mat_<cv::Point3f>> eta_plus;
    std::vector<cv::Mat_<cv::uchar>> cluster_plus;

    cv::Mat_<cv::Point3f> src_f_;
    cv::Mat_<cv::Point3f> src_joint_f_;

    cv::Mat_<cv::Point3f> sum_w_ki_Psi_blur_;
    cv::Mat_<float> sum_w_ki_Psi_blur_0_;

    cv::Mat_<float> min_pixel_dist_to_manifold_squared_;

    cv::RNG rng_;

    int cur_tree_height_;
    float sigma_r_over_sqrt_2_;

protected:
    double sigma_s_;
    double sigma_r_;
    int tree_height_;
    int num_pca_iterations_;

public:
    Adaptive_Manifold();
    ~Adaptive_Manifold();
    void changeint(){
        image = 10;
    }
    void display(){
        std::cout << Log2(2.0) << std::endl;
    }
    virtual void random_number_generation(cv::Mat & src, cv::Mat & src_joint, const cv::Size & srcSize);
    void apply_algorithm();

};

Adaptive_Manifold::Adaptive_Manifold(): Image_Preprocessing() {
    sigma_s_ = 16.0;
    sigma_r_ = 0.2;
    tree_height_ = -1;
    num_pca_iterations_ = 1;
}

Adaptive_Manifold::~Adaptive_Manifold() {
    eta_1.release();
    cluster_1.release();

    tilde_dst.release();
    alpha.release();
    diff.release();
    dst.release();

    V.release();

    dIcdx.release();
    dIcdy.release();
    dIdx.release();
    dIdy.release();
    dHdx.release();
    dVdy.release();

    t.release();

    theta_masked.release();
    mul.release();
    numerator.release();
    denominator.release();
    numerator_filtered.release();
    denominator_filtered.release();

    X.release();
    eta_k_small.release();
    eta_k_big.release();
    X_squared.release();
    pixel_dist_to_manifold_squared.release();
    gaussian_distance_weights.release();
    Psi_splat.release();
    Psi_splat_joined.release();
    Psi_splat_joined_resized.release();
    blurred_projected_values.release();
    w_ki_Psi_blur.release();
    w_ki_Psi_blur_0.release();
    w_ki_Psi_blur_resized.release();
    w_ki_Psi_blur_0_resized.release();
    rand_vec.release();
    v1.release();
    Nx_v1_mult.release();
    theta.release();

    eta_minus.clear();
    cluster_minus.clear();
    eta_plus.clear();
    cluster_plus.clear();

    src_f_.release();
    src_joint_f_.release();

    sum_w_ki_Psi_blur_.release();
    sum_w_ki_Psi_blur_0_.release();

    min_pixel_dist_to_manifold_squared_.release();
}
class MLAA : Image_Preprocessing {
public:
    //void apply_algorithm();
};

class classic_CV : Image_Preprocessing{
public:
    //void apply_algorithm();
};

class camera_driver : Image_Preprocessing {
public:
    //TODO: incorporate camera driver in here using either flycap2 or spinnaker
};


template <typename T>
void Image_Preprocessing::ensureSizeIsEnough(int rows, int cols, cv::Mat_<T>& m){
    if (m.empty() || m.data != m.datastart)
        m.create(rows, cols);
    else
    {
        const size_t esz = m.elemSize();
        const ptrdiff_t delta2 = m.dataend - m.datastart;

        const size_t minstep = m.cols * esz;

        cv::Size wholeSize;
        wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / m.step + 1), m.rows);
        wholeSize.width = std::max(static_cast<int>((delta2 - m.step * (wholeSize.height - 1)) / esz), m.cols);

        if (wholeSize.height < rows || wholeSize.width < cols)
            m.create(rows, cols);
        else
        {
            m.cols = cols;
            m.rows = rows;
        }
    }
}

template <typename T>
void Image_Preprocessing::ensureSizeIsEnough(const cv::Size size, cv::Mat_<T>& m){
    Image_Preprocessing::ensureSizeIsEnough(size.height, size.width, m);
}

template <typename T>
void Image_Preprocessing::h_filter(const cv::Mat_<T>& src, cv::Mat_<T>& dst, float sigma){
    //only runs during debug
    cv::CV_DbgAssert(src.depth() == cv::CV_32F);
    const float a = exp(-sqrt(2.0f) / sigma);
    ensureSizeIsEnough(src.size(), dst);
    src.copyTo(dst);
    for (int y = 0; y < src.rows; ++y){
        const T* src_row = src[y];
        T* dst_row = dst[y];
        for (int x = 1; x < src.cols; ++x)
        {
            dst_row[x] = src_row[x] + a * (src_row[x - 1] - src_row[x]);
        }
        for (int x = src.cols - 2; x >= 0; --x)
        {
            dst_row[x] = dst_row[x] + a * (dst_row[x + 1] - dst_row[x]);
        }
    }
    for (int y = 1; y < src.rows; ++y){
        T* dst_cur_row = dst[y];
        T* dst_prev_row = dst[y - 1];
        for (int x = 0; x < src.cols; ++x)
        {
            dst_cur_row[x] = dst_cur_row[x] + a * (dst_prev_row[x] - dst_cur_row[x]);
        }
    }
    for (int y = src.rows - 2; y >= 0; --y){
        T* dst_cur_row = dst[y];
        T* dst_prev_row = dst[y + 1];
        for (int x = 0; x < src.cols; ++x)
        {
            dst_cur_row[x] = dst_cur_row[x] + a * (dst_prev_row[x] - dst_cur_row[x]);
        }
    }
}

template<typename T>
void Image_Preprocessing::rdivide(const cv::Mat_<T>& a, const cv::Mat_<float>& b, cv::Mat_<T>& dst){
    //debug asserts
    cv::CV_DbgAssert(a.depth() == cv::CV_32F);
    cv::CV_DbgAssert(a.size() == b.size());

    ensureSizeIsEnough(a.size(), dst);
    dst.setTo(0);

    for (int y = 0; y < a.rows; ++y){
        const T* a_row = a[y];
        const float* b_row = b[y];
        T* dst_row = dst[y];

        for (int x = 0; x < a.cols; ++x){
            if (b_row[x] > std::numeric_limits<float>::epsilon())
                dst_row[x] = a_row[x] * (1.0f / b_row[x]);
        }
    }
}


//element wise multipication of two matrices - should have Point3f types
template <typename T>
void Image_Preprocessing::times(const cv::Mat_<T>& a, const cv::Mat_<float>& b, cv::Mat_<T>& dst){
    cv::CV_DbgAssert(a.depth() == cv::CV_32F);
    cv::CV_DbgAssert(a.size() == b.size());

    ensureSizeIsEnough(a.size(), dst);

    for (int y = 0; y < a.rows; ++y){
        const T* a_row = a[y];
        const float* b_row = b[y];
        T* dst_row = dst[y];

        for (int x = 0; x < a.cols; ++x){ //multiplying vectors of three elements not single numbers
            dst_row[x] = a_row[x] * b_row[x];
        }
    }
}

void Adaptive_Manifold::apply_algorithm(cv::Mat& src, cv::Mat& dst, cv::Mat& tilde_dst, cv::Mat& src_joint) {
    const cv::Size srcSize = src.size();
    cv::CV_Assert(src.type() == cv::CV_8UC3);
    cv::CV_Assert(src_joint.empty() || (src_joint.type() == src.type() && src_joint.size() == srcSize));
    Image_Preprocessing::ensureSizeIsEnough(srcSize, src_f_);
    src.convertTo(src_f_, src_f_.type(), 1.0 / 255.0);

    Adaptive_Manifold::random_number_generation(src, src_joint, const cv::Size & srcSize);
    //updating tree height
    cur_tree_height_ = tree_height_ > 0 ? tree_height_ : computeManifoldTreeHeight(sigma_s_, sigma_r_);

    // If no joint signal was specified, use the original signal
    ensureSizeIsEnough(srcSize, src_joint_f_);
    if (src_joint.empty())
        src_f_.copyTo(src_joint_f_);
    else
        src_joint.convertTo(src_joint_f_, src_joint_f_.type(), 1.0 / 255.0);

    // Dividing the covariance matrix by 2 is equivalent to dividing the standard deviations by sqrt(2).
    sigma_r_over_sqrt_2_ = static_cast<float>(sigma_r_ / sqrt(2.0));

    //Compute first manifold via low pass filtering
    h_filter(src_joint_f_, eta_1, static_cast<float>(sigma_s_));

    ensureSizeIsEnough(srcSize, cluster_1);
    cluster_1.setTo(Scalar::all(1));

    eta_minus.resize(cur_tree_height_);
    cluster_minus.resize(cur_tree_height_);
    eta_plus.resize(cur_tree_height_);
    cluster_plus.resize(cur_tree_height_);
    buildManifoldsAndPerformFiltering(eta_1, cluster_1, 1); //TODO: create this function

    // Compute the filter response by normalized convolution -- Eq. (4)
    rdivide(sum_w_ki_Psi_blur_, sum_w_ki_Psi_blur_0_, tilde_dst);

    // Adjust the filter response for outlier pixels -- Eq. (10)
    ensureSizeIsEnough(srcSize, alpha);
    cv::exp(min_pixel_dist_to_manifold_squared_ * (-0.5 / sigma_r_ / sigma_r_), alpha);

    ensureSizeIsEnough(srcSize, diff);
    cv::subtract(tilde_dst, src_f_, diff);
    times(diff, alpha, diff);

    ensureSizeIsEnough(srcSize, dst);
    cv::add(src_f_, diff, dst);

    /*
    dst.convertTo(dst, cv::CV_8U, 255.0);
    if (tilde_dst.needed())
        tilde_dst.convertTo(tilde_dst, cv::CV_8U, 255.0);
    */
}

/*
 * Summary: Determines random number starting from center pixel
 */

virtual void Adaptive_Manifold::random_number_generation(cv::Mat & src, cv::Mat & src_joint, const cv::Size & srcSize) {
    // Use the center pixel as seed to random number generation.
    const cv::Point3f centralPix = src_f_(src_f_.rows / 2, src_f_.cols / 2);
    const double seedCoeff = (centralPix.x + centralPix.y + centralPix.z + 1.0f) / 4.0f;
    rng_.state = static_cast<cv::uint64>(seedCoeff * numeric_limits<cv::uint64>::max());

    ensureSizeIsEnough(srcSize, sum_w_ki_Psi_blur_);
    sum_w_ki_Psi_blur_.setTo(cv::Scalar::all(0));

    ensureSizeIsEnough(srcSize, sum_w_ki_Psi_blur_0_);
    sum_w_ki_Psi_blur_0_.setTo(cv::Scalar::all(0));

    ensureSizeIsEnough(srcSize, min_pixel_dist_to_manifold_squared_);
    min_pixel_dist_to_manifold_squared_.setTo(cv::Scalar::all(std::numeric_limits<float>::max()));
}
#endif //IMAGE_PREPROCESSING_PIPELINE_ADAPTIVE_MANIFOLD_HPP

void Image_Preprocessing::channelsSum(const Mat_<cv::Point3f>& src, cv::Mat_<float>& dst){
    ensureSizeIsEnough(src.size(), dst);
    for (int y = 0; y < src.rows; ++y)
    {
        const cv::Point3f* src_row = src[y];
        float* dst_row = dst[y];
        for (int x = 0; x < src.cols; ++x)
        {
            const cv::Point3f src_val = src_row[x];
            dst_row[x] = src_val.x + src_val.y + src_val.z;
        }
    }
}

void Image_Preprocessing::phi(const cv::Mat_<float>& src, cv::Mat_<float>& dst, float sigma){
    ensureSizeIsEnough(src.size(), dst);
    for (int y = 0; y < dst.rows; ++y)
    {
        const float* src_row = src[y];
        float* dst_row = dst[y];
        for (int x = 0; x < dst.cols; ++x)
        {
            dst_row[x] = exp(-0.5f * src_row[x] / sigma / sigma);
        }
    }
}

void Image_Preprocessing::catCn(const Mat_<cv::Point3f>& a, const cv::Mat_<float>& b, Mat_<cv::Vec4f>& dst)
{
    ensureSizeIsEnough(a.size(), dst);
    for (int y = 0; y < a.rows; ++y)
    {
        const cv::Point3f* a_row = a[y];
        const float* b_row = b[y];
        cv::Vec4f* dst_row = dst[y];
        for (int x = 0; x < a.cols; ++x)
        {
            const cv::Point3f a_val = a_row[x];
            const float b_val = b_row[x];
            dst_row[x] = cv::Vec4f(a_val.x, a_val.y, a_val.z, b_val);
        }
    }
}