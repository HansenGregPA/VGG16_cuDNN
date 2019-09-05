#include "util.h"

cv::Mat load_image(const char* image_path)
{
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	image.convertTo(image, CV_32FC3);
	//cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

void save_image(std::string output_filename, float *buffer, int h, int w)
{
	cv::Mat output_image(h, w, CV_32FC3, buffer);
	cv::threshold(output_image, output_image, 0, 0, cv::THRESH_TOZERO);
	output_image.convertTo(output_image, CV_8UC3);
	cv::imwrite(output_filename, output_image);
}