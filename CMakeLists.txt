cmake_minimum_required(VERSION 3.0.0)
project(ComputerVisionCuda )

include(CTest)
enable_testing()


find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_22,code=sm_22
    )

include_directories(${OPENCV_INCLUDE_DIRS})

# Specify include directories
include_directories(
    kernels
    utility
    )
# Specify library paths
link_directories(
    /opt/foobar/lib
    /opt/joestuff/lib
    )



add_executable(${PROJECT_NAME} src/cpu/main.cpp src/cpu/CompVis.cpp)



# For compilation ...
# Specify target & source files to compile it from
# cuda_add_executable(
#     ${PROJECT_NAME}
#     kernel.cu
#     hellocuda.h
#     kernels/hellokernels.cu
#     kernels/hellokernels.h
#     utility/wrapper.cpp
#     utility/wrapper.h
#     )

# if(OpenMP_CXX_FOUND)
#     target_link_libraries(MyTarget PUBLIC OpenMP::OpenMP_CXX)
# endif()

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
# For linking ...
# Specify target & libraries to link it with
# target_link_libraries(
#     hellocuda
#     -lfoobar
#     -ljoestuff
#     )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)