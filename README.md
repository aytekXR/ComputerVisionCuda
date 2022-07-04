# CUDA OpenMP Single Thread Benchmarking Based On Computer Vision Pipeline 

## About <a name = "about"></a>

In this project, I have implemented a vanilla vision pipeline which consists of convolution batch normalization and relu. Based on this plan, the pipeline is coded from scratch for each of the programing framework.

Due to incompatibilities CUDA part is implemented on msvs22, while openmp and single thread implementations are made on linux.




## Getting Started <a name = "getting_started"></a>

If you would like to have a quick test, You can use appropriate prebuilts under given directory. A video demo is for each as:
 --[windows-cuda](https://youtu.be/wUx9cP_RaSI)
 --[linux-openmp](https://youtu.be/aI6w3pGCmo4)

### Prerequisites

In order to use this repo you should have:
```
cuda-toolkit, opencv, openmp
```

### Installing

A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
git clone https://github.com/aytekine/ComputerVisionCuda.git
mkdir build; cd build
cmake .. .
make
```

After a succesfull build you should be capable of using the binary as giving a /path/to/the/ imagedir as:
```
ComputerVisionCuda_openmp ../inputImageDir
```

End with an example of getting some data out of the system or using it for a little demo.