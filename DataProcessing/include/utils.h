//
// Created by iceytan on 18-11-30.
//

#ifndef SOURCE_GENERATEIMG_UTILS_H
#define SOURCE_GENERATEIMG_UTILS_H
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_SILENCE_DEPRECATION

#include <cl2.hpp>

using namespace cl;

void pbar(int cur, int total);

std::pair<Context, Device> CreateGPUContextDevice();


#endif //SOURCE_GENERATEIMG_UTILS_H
