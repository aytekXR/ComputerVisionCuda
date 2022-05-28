#pragma once
#include"CompVis.h"
#include <chrono>
#include <thread> 
#include <iostream>

/*
CompVis myObj("./inputImageDir/polling/*.jpg", 512, 512, 0);

void startPreprocess()
{
	std::cout << "Start of Preprocess"<<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	myObj.init();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time ("<<NUM_OF_THREAT<<" threads) for Preprocess!: " << elapsed.count() << " s\n\n";

}

void prepareKernels()
{
	std::vector<float> kernel1{ -1,0,1,-2,0,2,-1,0,1 };
	std::vector<float> kernel3(9, 0.11);

#if DebugMod
	std::vector<float> kernel2(8, 0.11);
	myObj.addKernel(kernel2);
#endif
	myObj.addKernel(kernel3);
	myObj.addKernel(kernel1);
	myObj.addKernel({ -1,-2,-1,0,0,0,1,2,1 });
}


void startConvolutions()
{
	std::cout << "Start of Convolutions!" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	myObj.convolveMemberLists();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time (" << NUM_OF_THREAT << " threads) for Convolutions!: " << elapsed.count() << " s\n\n";
}

void startBatchNormalization()
{
	
	std::cout << "Start of Batch Normalization!" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	myObj.batchNormalize();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time (" << NUM_OF_THREAT << " threads) for Batch Normalization!: " << elapsed.count() << " s\n\n";
}

void startRelu()
{
	std::this_thread::sleep_for(std::chrono::seconds(1));
	std::cout << "Start of ReLU" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	myObj.relu();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time (" << NUM_OF_THREAT << " threads) for ReLU!: " << elapsed.count() << " s\n\n";

}


void showOverallResults()
{
	myObj.visualizeOutputs();
}

void RunAllConsecutiveSteps()
{
	startPreprocess();
	prepareKernels();
	startConvolutions();
	startBatchNormalization();
	startRelu();
	showOverallResults();
}
*/

