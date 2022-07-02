cmake_minimum_required(VERSION 3.0.0)
project(ComputerVisionCuda )

include(CTest)
enable_testing()


find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp CompVis.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
