#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "adaptive_manifold.hpp"

int main(int argc, const char* argv[])
{
	const string inputImageName = //cmd.get<string>("input");
	const string outputImageName = cmd.get<string>("output");
	const string jointImageName = cmd.get<string>("joint");
	const double sigma_s = cmd.get<double>("sigma_s");
	const double sigma_r = cmd.get<double>("sigma_r");
	const int tree_height = cmd.get<int>("tree_height");
	const int num_pca_iterations = cmd.get<int>("num_pca_iterations");

	if (inputImageName.empty())
	{
		cerr << "Missing input image" << endl;
		cmd.printParams();
		return -1;
	}

    cv::Mat img = imread(inputImageName);
	if (img.empty())
	{
		cerr << "Can't open image - " << inputImageName << endl;
		return -1;
	}

    cv::Mat jointImg;
	if (!jointImageName.empty())
	{
		jointImg = imread(jointImageName);
		if (jointImg.empty())
		{
			cerr << "Can't open image - " << inputImageName << endl;
			return -1;
		}
	}

	Ptr<AdaptiveManifoldFilter> filter = AdaptiveManifoldFilter::create();
	filter->set("sigma_s", sigma_s);
	filter->set("sigma_r", sigma_r);
	filter->set("tree_height", tree_height);
	filter->set("num_pca_iterations", num_pca_iterations);

    cv::Mat dst, tilde_dst;
	filter->apply(img, dst, tilde_dst, jointImg);

	if (!outputImageName.empty())
	{
		const string::size_type dotPos = outputImageName.find_last_of('.');
		const string name = outputImageName.substr(0, dotPos);
		const string ext = outputImageName.substr(dotPos + 1);

		imwrite(outputImageName, dst);
		imwrite(name + "_tilde." + ext, tilde_dst);
	}

	imshow("Input", img);
	imshow("Output", dst);
	imshow("Tilde Output", tilde_dst);
	imshow("NLM", nlm_dst);
	waitKey();

	return 0;
}