#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>

#define DebugMod true 
#define NUM_OF_THREAT 1

class CompVis
{

public:
	CompVis(cv::String ImgFolderPath, uint16_t imageWidth, uint16_t imageHight, uint16_t ImageNum);
	void init();

public: // preproces
	std::vector<cv::Mat_<float>> readAndPreprocess();
	

public: // convolution layer
	void addKernel(std::vector<float> singlekernel);
	cv::Mat_<float> singleImageSingleKernelConv(const cv::Mat_<float>& src, const std::vector<float>& kernel);
	void convolveLists(std::vector<std::vector<float>>& kernelListW);
	void convolveMemberLists();

public: // batch normalization 
	cv::Mat_<float> singleImageMeanVariance(const cv::Mat_<float>& src);



public:
	cv::String m_pattern;
	std::vector<uint16_t> m_imgSize;
	uint16_t m_numofImages;
	std::vector<int> m_kernelSize{ 3,3};

public:
	std::vector<cv::Mat_<float>> m_batchListX; //preprocess output
	std::vector<std::vector<float>> m_kernelListW;
	std::vector<cv::Mat_<float>> m_batchListYwantedForm; //convolution outputs v1  form: BxHxWxC
	std::vector<std::vector<cv::Mat_<float>>> m_batchListYusefullForm; //convolution outputs v2 form: BxCxHxW

};