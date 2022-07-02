#include "CompVis.h"
#include<iostream>

CompVis::CompVis(cv::String ImgFolderPath, uint16_t imageWidth, uint16_t imageHight, uint16_t ImageNum = 0)
{
	m_pattern = ImgFolderPath;
	m_imgSize.emplace_back(ImageNum);
	m_imgSize.emplace_back(imageWidth);
	m_imgSize.emplace_back(imageHight);

}

void CompVis::init()
{
	m_batchListX = readAndPreprocess();
#if DebugMod
	std::cout << "END OF readAndPreprocess" << std::endl;
	for (cv::Mat_<float> img : m_batchListX) {
		cv::imshow("Raw Image", img / 256);
		cv::waitKey(0);
	}
#endif

}

std::vector<cv::Mat_<float>> CompVis::readAndPreprocess()
{
	std::vector<cv::String> filenames;
	cv::glob(m_pattern, filenames, false);

	m_imgSize[0] == 0 ? m_numofImages = filenames.size() : m_numofImages = m_imgSize[0];

	std::vector<cv::Mat> images(m_numofImages);
	std::vector<cv::Mat>::iterator imgPtr = images.begin();

#pragma omp parallel for num_threads(NUM_OF_THREAT)
	for (int imgPath = 0; imgPath < m_numofImages; imgPath++)
	{
		//resize(cv::imread(filenames[imgPath], cv::IMREAD_GRAYSCALE), *imgPtr++, cv::Size(m_imgSize[1], m_imgSize[2]), cv::INTER_LINEAR); //not thread safe
		resize(cv::imread(filenames[imgPath], cv::IMREAD_GRAYSCALE), images[imgPath], cv::Size(m_imgSize[1], m_imgSize[2]), cv::INTER_LINEAR); //openmp competible
	}

	// use copy semantic to change Mat objects to Mat_<float>
	std::vector<cv::Mat_<float>> m_vcopy(images.size());
	copy(images.begin(), images.end(), m_vcopy.begin());

	return m_vcopy;
}

void CompVis::addKernel(std::vector<float> singlekernel)
{
    try {
        if (singlekernel.size() == m_kernelSize[0]* m_kernelSize[1]) {
            m_kernelListW.emplace_back(singlekernel);
#if DebugMod
            std::cout << "Kernel Added to the member array!" << std::endl;
#endif
        }
        else {
            throw ((int)singlekernel.size());
        }
    }
    catch (int errN) {
        std::cout << "Kernel Size does not match! Could not add Kernel!"<<
            m_kernelSize[0] * m_kernelSize[1]<< " element are required!" <<
            errN<< " are given!"<< std::endl;
    }
}
cv::Mat_<float> CompVis::singleImageSingleKernelConv(const cv::Mat_<float>& src, const std::vector<float>& kernel)
{
    cv::Mat_<float> convOut(cv::Size(src.rows - m_kernelSize[0] + 1, src.cols - m_kernelSize[1] + 1));
    cv::Mat_<float>::iterator outPtr = convOut.begin();
    //cv::Mat_<float>::iterator inPtr = src.begin();

//#pragma omp parallel for num_threads(NUM_OF_THREAT)
    for (int i = 0; i < src.rows - m_kernelSize[0] +1; i++) {// src rows
       // inPtr = src.ptr<float>(i);
        for (int j = 0; j < src.cols - m_kernelSize[1] +1; j++) {// src cols

            float tmp = 0;
            int kernelIndex = 0;
            for (int k = 0; k < m_kernelSize[0]; k++) {// kernel rows
                for (int l = 0; l < m_kernelSize[1]; l++) {// kernel cols
                    tmp += src.at<float>(i+k, j+l) * kernel[kernelIndex++]; //Alternative 1
                    //tmp += src[i + k, j + l] * kernel[kernelIndex++]; //Alternative 2
                    //tmp += (*(inPtr+k*src.rows+j+l)) * kernel[kernelIndex++]; //Alternative 3
                }
            }
            //inPtr++;
            *outPtr++ = tmp;
        }
    }
    return convOut.clone();
}

//std::vector<std::vector<cv::Mat_<float>>> CompVis::convolveLists(std::vector<std::vector<float>>& kernelListW)
void CompVis::convolveLists(std::vector<std::vector<float>>& kernelListW)
{
    //m_batchListYwantedForm.reserve(m_numofImages);
    //m_batchListYusefullForm.reserve(m_numofImages);

#pragma omp parallel for num_threads(NUM_OF_THREAT)
    for (int imgIndex = 0; imgIndex < m_numofImages; imgIndex++) { //for each image in list X, use openmp

        std::vector<cv::Mat_<float>> convOut4SingImg;


        for (std::vector<float>& kernel : kernelListW) { //for each kernel in the set of W
            cv::Mat_<float> tmp = singleImageSingleKernelConv(m_batchListX[imgIndex], kernel);
            convOut4SingImg.push_back(tmp);
        }

        //m_batchListYusefullForm[imgIndex] = convOut4SingImg;
        m_batchListYusefullForm.emplace_back(convOut4SingImg);


        cv::Mat tmp2(cv::Size(m_imgSize[1] - m_kernelSize[0] + 1, m_imgSize[2] - m_kernelSize[1] + 1), CV_32FC(kernelListW.size()));
        cv::merge(convOut4SingImg, tmp2);
        m_batchListYwantedForm.emplace_back(tmp2);

    }
}

void CompVis::convolveMemberLists()
{
    convolveLists(m_kernelListW);
}


cv::Mat_<float> CompVis::singleImageMeanVariance(const cv::Mat_<float>& src)
{
    double sum = std::accumulate(src.begin(), src.end(), 0.0);
    double mean = sum / (src.rows*src.cols);

    double sq_sum = std::inner_product(src.begin(), src.end(), src.begin(), 0.0);
    double var =   (sq_sum / (src.rows * src.cols) - mean * mean);
    double stdev = std::sqrt(var);


    cv::Mat_<float> dst = src.clone();
    
    
    dst -= mean;
    dst /= var;
    //dst /= stdev; // when normalize to zero mean unit varience, image is getting too much darker. Used for this purpose

#if DebugMod
    cv::imshow("SRC", src/256);
    cv::imshow("DST", dst);
    cv::waitKey(0);
    std::cout << "Mean and Variance are " << mean << " " << var << std::endl;
#endif


    return dst;
}


