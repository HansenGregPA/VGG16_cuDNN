#ifndef _UTIL_H_
#define _UTIL_H_

#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>

cv::Mat load_image(const char* image_path);

void save_image(std::string output_filename, float *buffer, int h, int w);

#endif // !_UTIL_H_
#pragma once
