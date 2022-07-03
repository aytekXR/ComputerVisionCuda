#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>

#define DebugMod true 
#define NUM_OF_THREAT 20

class CompVis
{

public:
	CompVis(cv::String ImgFolderPath, uint16_t imageWidth, uint16_t imageHight, uint16_t ImageNum);

public: // preproces
	void init();

private:
	std::vector<cv::Mat_<float>> readAndPreprocess();
	

public: // convolution layer
	void addKernel(std::vector<float> singlekernel);
	void convolveMemberLists();

private:
	cv::Mat_<float> singleImageSingleKernelConv(const cv::Mat_<float>& src, const std::vector<float>& kernel);
	void convolveLists(std::vector<std::vector<float>>& kernelListW);

public: // batch normalization 
	void batchNormalize();

private:
	std::vector<double> singleImageMeanVariance(const cv::Mat_<float>& src);
	cv::Mat_<float> batchNormSingleImage(const cv::Mat_<float>& src, double mean, double var);

public: // ReLU
	void relu();
	void visualizeOutputs();

private:
	void reluSingleImage(cv::Mat_<float>& src);

private:
	cv::String m_pattern;
	std::vector<uint16_t> m_imgSize;
	uint16_t m_numofImages;
	std::vector<int> m_kernelSize{ 3,3};
	std::vector<double> m_means;
	std::vector<double> m_vars;

private:
	std::vector<cv::Mat_<float>> m_batchListX; //preprocess output
	std::vector<std::vector<float>> m_kernelListW;
	std::vector<cv::Mat_<float>> m_batchListYwantedForm; //convolution outputs v1  form: BxHxWxC
	std::vector<std::vector<cv::Mat_<float>>> m_batchListYusefullForm; //convolution outputs v2 form: BxCxHxW
	std::vector<cv::Mat_<float>> m_batchListZwantedForm; //batch normalizaton outputs v1  form: BxHxWxC
	std::vector<std::vector<cv::Mat_<float>>> m_batchListZusefullForm; //batch normalizaton outputs v2 form: BxCxHxW
	std::vector<cv::Mat_<float>> m_batchListVwantedForm; //ReLU outputs v1  form: BxHxWxC
	std::vector<std::vector<cv::Mat_<float>>> m_batchListVusefullForm; // ReLU outputs v2 form: BxCxHxW
};