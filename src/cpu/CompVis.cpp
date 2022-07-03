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
        int tmpFilterSize = m_kernelSize[0] * m_kernelSize[1];
        if (singlekernel.size() == tmpFilterSize) {
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

void CompVis::convolveLists(std::vector<std::vector<float>>& kernelListW)
{

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


std::vector<double> CompVis::singleImageMeanVariance(const cv::Mat_<float>& src)
{
    std::vector<double> meanVar;
    double sum = std::accumulate(src.begin(), src.end(), 0.0);
    double mean = sum / (src.rows*src.cols);

    double sq_sum = std::inner_product(src.begin(), src.end(), src.begin(), 0.0);
    double var =   (sq_sum / (src.rows * src.cols) - mean * mean);
    
    meanVar.push_back(mean);
    meanVar.push_back(sq_sum);

    return meanVar;
}


cv::Mat_<float> CompVis::batchNormSingleImage(const cv::Mat_<float>& src, double mean, double var)
{
    cv::Mat_<float> dst = src.clone();

    dst -= mean;
    dst /= var;

#if DebugMod
    cv::imshow("SRC", src / 256);
    cv::imshow("DST", dst/256);
    cv::waitKey(0);
    std::cout << "Mean and Variance are " << mean << " " << var << std::endl;
#endif

    return dst;
}

void CompVis::batchNormalize()
{
    for (int channel = 0; channel < m_batchListYusefullForm[0].size(); channel ++) {
        long double means = 0;
        long double sq_sum = 0;
        long int Imgcounter = 0;
#pragma omp parallel for num_threads(NUM_OF_THREAT)
        for (int batch = 0; batch < m_batchListYusefullForm.size(); batch++) {

            std::vector<double> tmp = singleImageMeanVariance(m_batchListYusefullForm[batch][channel]);
            means  += tmp[0];
            sq_sum += tmp[1];
            Imgcounter++;

        }

        int numOfPixelsPerImage = (m_imgSize[1] - m_kernelSize[0]+1) * (m_imgSize[2] - m_kernelSize[1]+1);
        double mean = means / Imgcounter;

        Imgcounter *= (numOfPixelsPerImage * m_batchListYusefullForm.size());
        double var = (sq_sum / Imgcounter - mean * mean);
        m_means.push_back(mean);
        m_vars.push_back(var);
        means = 0;
        sq_sum = 0;
        Imgcounter = 0;
    }


//#pragma omp parallel for num_threads(NUM_OF_THREAT)
    for (int batch = 0; batch < m_batchListYusefullForm.size(); batch++) {
        std::vector<cv::Mat_<float>> normOut4SingImg;
        for (int channel = 0; channel < m_batchListYusefullForm[0].size(); channel++) {
            cv::Mat_<float> tmp = batchNormSingleImage(m_batchListYusefullForm[batch][channel], m_means[channel],m_vars[channel]);
            normOut4SingImg.push_back(tmp);
        }
        m_batchListZusefullForm.push_back(normOut4SingImg);
        cv::Mat tmp2(cv::Size(m_imgSize[1] - m_kernelSize[0] + 1, m_imgSize[2] - m_kernelSize[1] + 1), CV_32FC(m_kernelListW.size()));
        cv::merge(normOut4SingImg, tmp2);
        m_batchListZwantedForm.push_back(tmp2);
    }
}

void CompVis::reluSingleImage(cv::Mat_<float>& src)
{
    for (cv::Mat_<float>::iterator PixelIter = src.begin(); PixelIter != src.end(); PixelIter++) {

        *PixelIter = std::max((float)0.0, *PixelIter);
    }
#if DebugMod
    cv::imshow("Output of ReLU", src);
#endif
}

void CompVis::relu()
{
    m_batchListVusefullForm = m_batchListZusefullForm;
//#pragma omp parallel for num_threads(NUM_OF_THREAT)
    for (int batch = 0; batch < m_batchListVusefullForm.size(); batch++) {
        std::vector<cv::Mat_<float>> reluOut4SingImg;

        for (int channel = 0; channel < m_batchListVusefullForm[batch].size(); channel++) {
            reluSingleImage(m_batchListVusefullForm[batch][channel]);
        }

        cv::Mat tmp2(cv::Size(m_imgSize[1] - m_kernelSize[0] + 1, m_imgSize[2] - m_kernelSize[1] + 1), CV_32FC(m_kernelListW.size()));
        cv::merge(m_batchListVusefullForm[batch], tmp2);
        m_batchListVwantedForm.emplace_back(tmp2);
    }
}

void CompVis::visualizeOutputs()
          {
#if DebugMod
    for (int batch = 0; batch < m_batchListVusefullForm.size(); batch++) {
        for (int channel = 0; channel < m_batchListVusefullForm[batch].size(); channel++) {

            std::string message = "ReLu Output for Batch: " + std::to_string(batch) + " Channel: " + std::to_string(channel);
            std::string imgName = "./outputImageDir/B" + std::to_string(batch) + "C" + std::to_string(channel) + ".jpg";

            if (channel == 0) {
                cv::imshow(message, m_batchListVusefullForm[batch][channel]*256);
                cv::imwrite(imgName,m_batchListVusefullForm[batch][channel] * 256);
            }
            else {
                cv::imshow(message, m_batchListVusefullForm[batch][channel]);
                cv::imwrite(imgName,m_batchListVusefullForm[batch][channel]);
            }

            cv::waitKey(0);

        }

    }
#endif
}
