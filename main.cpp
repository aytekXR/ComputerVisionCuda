/*
#include <iostream>

int main(int, char**) {
    std::cout << "Hello, world!\n";
}

*/

#include"CompVis.h"

#include<iostream>



int main()
{
	CompVis myObj("/home/ae/repo/ComputerVisionCuda/inputImageDir/*.jpg",512,512,0);

	myObj.init();


	std::vector<float> kernel0(8, 0.11);
	std::vector<float> kernel2{ -1,0,1,-2,0,2,-1,0,1 };
	std::vector<float> kernel3(9, 0.11);
	std::vector<std::vector<float>> kernelListW;

	myObj.addKernel(kernel0);
	myObj.addKernel(kernel2);
	myObj.addKernel(kernel3);
	myObj.addKernel({ -1,-2,-1,0,0,0,1,2,1 });

    //myObj.convolveLists(kernelListW);
	//myObj.convolveLists(myObj.m_kernelListW);
	//myObj.convolveMemberLists();

	for (int i = 0; i < 12; i++) {
		myObj.singleImageMeanVariance(myObj.m_batchListX[i]);
	}

	return 0;
}
