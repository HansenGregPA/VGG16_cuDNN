#include <iostream>
#include <string>
#include <vector>
#include <opencv2\opencv.hpp>
#include <cudnn.h>
#include "vgg16.h"
#include "util.h"
#include "npy.hpp"

using namespace std;

int main(/*int argc, const char* argv[]*/) 
{
	//load parameters
	//定义npy参数文件路径
	string base_path = "../vgg16/";
	string file_path = "";

	vector<std::vector<float>> conv_kernels;
	vector<std::vector<float>> conv_bias;

	//定义变量名
	vector<unsigned long> conv1_1_weights_shape;
	std::vector<float> conv1_1_weights_data;
	vector<unsigned long> conv1_1_biases_shape;
	std::vector<float> conv1_1_biases_data;

	vector<unsigned long> conv1_2_weights_shape;
	std::vector<float> conv1_2_weights_data;
	vector<unsigned long> conv1_2_biases_shape;
	std::vector<float> conv1_2_biases_data;

	vector<unsigned long> conv2_1_weights_shape;
	std::vector<float> conv2_1_weights_data;
	vector<unsigned long> conv2_1_biases_shape;
	std::vector<float> conv2_1_biases_data;

	vector<unsigned long> conv2_2_weights_shape;
	std::vector<float> conv2_2_weights_data;
	vector<unsigned long> conv2_2_biases_shape;
	std::vector<float> conv2_2_biases_data;

	vector<unsigned long> conv3_1_weights_shape;
	std::vector<float> conv3_1_weights_data;
	vector<unsigned long> conv3_1_biases_shape;
	std::vector<float> conv3_1_biases_data;

	vector<unsigned long> conv3_2_weights_shape;
	std::vector<float> conv3_2_weights_data;
	vector<unsigned long> conv3_2_biases_shape;
	std::vector<float> conv3_2_biases_data;

	vector<unsigned long> conv3_3_weights_shape;
	std::vector<float> conv3_3_weights_data;
	vector<unsigned long> conv3_3_biases_shape;
	std::vector<float> conv3_3_biases_data;

	vector<unsigned long> conv4_1_weights_shape;
	std::vector<float> conv4_1_weights_data;
	vector<unsigned long> conv4_1_biases_shape;
	std::vector<float> conv4_1_biases_data;

	vector<unsigned long> conv4_2_weights_shape;
	std::vector<float> conv4_2_weights_data;
	vector<unsigned long> conv4_2_biases_shape;
	std::vector<float> conv4_2_biases_data;

	vector<unsigned long> conv4_3_weights_shape;
	std::vector<float> conv4_3_weights_data;
	vector<unsigned long> conv4_3_biases_shape;
	std::vector<float> conv4_3_biases_data;

	vector<unsigned long> conv5_1_weights_shape;
	std::vector<float> conv5_1_weights_data;
	vector<unsigned long> conv5_1_biases_shape;
	std::vector<float> conv5_1_biases_data;

	vector<unsigned long> conv5_2_weights_shape;
	std::vector<float> conv5_2_weights_data;
	vector<unsigned long> conv5_2_biases_shape;
	std::vector<float> conv5_2_biases_data;

	vector<unsigned long> conv5_3_weights_shape;
	std::vector<float> conv5_3_weights_data;
	vector<unsigned long> conv5_3_biases_shape;
	std::vector<float> conv5_3_biases_data;

	vector<unsigned long> fc6_weights_shape;
	std::vector<float> fc6_weights_data;
	vector<unsigned long> fc6_biases_shape;
	std::vector<float> fc6_biases_data;

	vector<unsigned long> fc7_weights_shape;
	std::vector<float> fc7_weights_data;
	vector<unsigned long> fc7_biases_shape;
	std::vector<float> fc7_biases_data;

	vector<unsigned long> fc8_weights_shape;
	std::vector<float> fc8_weights_data;
	vector<unsigned long> fc8_biases_shape;
	std::vector<float> fc8_biases_data;

	vector<unsigned long> mean_rgb_shape;
	std::vector<float> mean_rgb_data;

	// ****************************************************************************
	//load conv parameters
	file_path = base_path + "vgg_16-conv1-conv1_1-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv1_1_weights_shape, conv1_1_weights_data);
	file_path = base_path + "vgg_16-conv1-conv1_1-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv1_1_biases_shape, conv1_1_biases_data);
	// 
	conv_kernels.push_back(conv1_1_weights_data);
	conv_bias.push_back(conv1_1_biases_data);

	file_path = base_path + "vgg_16-conv1-conv1_2-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv1_2_weights_shape, conv1_2_weights_data);
	file_path = base_path + "vgg_16-conv1-conv1_2-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv1_2_biases_shape, conv1_2_biases_data);
	// 
	conv_kernels.push_back(conv1_2_weights_data);
	conv_bias.push_back(conv1_2_biases_data);

	file_path = base_path + "vgg_16-conv2-conv2_1-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv2_1_weights_shape, conv2_1_weights_data);
	file_path = base_path + "vgg_16-conv2-conv2_1-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv2_1_biases_shape, conv2_1_biases_data);
	// 
	conv_kernels.push_back(conv2_1_weights_data);
	conv_bias.push_back(conv2_1_biases_data);

	file_path = base_path + "vgg_16-conv2-conv2_2-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv2_2_weights_shape, conv2_2_weights_data);
	file_path = base_path + "vgg_16-conv2-conv2_2-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv2_2_biases_shape, conv2_2_biases_data);
	// 
	conv_kernels.push_back(conv2_2_weights_data);
	conv_bias.push_back(conv2_2_biases_data);

	file_path = base_path + "vgg_16-conv3-conv3_1-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv3_1_weights_shape, conv3_1_weights_data);
	file_path = base_path + "vgg_16-conv3-conv3_1-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv3_1_biases_shape, conv3_1_biases_data);
	// 
	conv_kernels.push_back(conv3_1_weights_data);
	conv_bias.push_back(conv3_1_biases_data);

	file_path = base_path + "vgg_16-conv3-conv3_2-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv3_2_weights_shape, conv3_2_weights_data);
	file_path = base_path + "vgg_16-conv3-conv3_2-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv3_2_biases_shape, conv3_2_biases_data);
	// 
	conv_kernels.push_back(conv3_2_weights_data);
	conv_bias.push_back(conv3_2_biases_data);

	file_path = base_path + "vgg_16-conv3-conv3_3-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv3_3_weights_shape, conv3_3_weights_data);
	file_path = base_path + "vgg_16-conv3-conv3_3-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv3_3_biases_shape, conv3_3_biases_data);
	// 
	conv_kernels.push_back(conv3_3_weights_data);
	conv_bias.push_back(conv3_3_biases_data);

	file_path = base_path + "vgg_16-conv4-conv4_1-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv4_1_weights_shape, conv4_1_weights_data);
	file_path = base_path + "vgg_16-conv4-conv4_1-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv4_1_biases_shape, conv4_1_biases_data);
	// 
	conv_kernels.push_back(conv4_1_weights_data);
	conv_bias.push_back(conv4_1_biases_data);

	file_path = base_path + "vgg_16-conv4-conv4_2-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv4_2_weights_shape, conv4_2_weights_data);
	file_path = base_path + "vgg_16-conv4-conv4_2-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv4_2_biases_shape, conv4_2_biases_data);
	// 
	conv_kernels.push_back(conv4_2_weights_data);
	conv_bias.push_back(conv4_2_biases_data);

	file_path = base_path + "vgg_16-conv4-conv4_3-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv4_3_weights_shape, conv4_3_weights_data);
	file_path = base_path + "vgg_16-conv4-conv4_3-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv4_3_biases_shape, conv4_3_biases_data);
	// 
	conv_kernels.push_back(conv4_3_weights_data);
	conv_bias.push_back(conv4_3_biases_data);

	file_path = base_path + "vgg_16-conv5-conv5_1-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv5_1_weights_shape, conv5_1_weights_data);
	file_path = base_path + "vgg_16-conv5-conv5_1-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv5_1_biases_shape, conv5_1_biases_data);
	// 
	conv_kernels.push_back(conv5_1_weights_data);
	conv_bias.push_back(conv5_1_biases_data);

	file_path = base_path + "vgg_16-conv5-conv5_2-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv5_2_weights_shape, conv5_2_weights_data);
	file_path = base_path + "vgg_16-conv5-conv5_2-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv5_2_biases_shape, conv5_2_biases_data);
	// 
	conv_kernels.push_back(conv5_2_weights_data);
	conv_bias.push_back(conv5_2_biases_data);

	file_path = base_path + "vgg_16-conv5-conv5_3-weights.npy";
	npy::LoadArrayFromNumpy(file_path, conv5_3_weights_shape, conv5_3_weights_data);
	file_path = base_path + "vgg_16-conv5-conv5_3-biases.npy";
	npy::LoadArrayFromNumpy(file_path, conv5_3_biases_shape, conv5_3_biases_data);
	// 
	conv_kernels.push_back(conv5_3_weights_data);
	conv_bias.push_back(conv5_3_biases_data);
	// ****************************************************************************

	// ****************************************************************************
	vector<std::vector<float>> fc_paras;
	vector<std::vector<float>> fc_bias;
	// load fully connected parameters
	file_path = base_path + "vgg_16-fc6-weights.npy";
	npy::LoadArrayFromNumpy(file_path, fc6_weights_shape, fc6_weights_data);
	file_path = base_path + "vgg_16-fc6-biases.npy";
	npy::LoadArrayFromNumpy(file_path, fc6_biases_shape, fc6_biases_data);
	fc_paras.push_back(fc6_weights_data);
	fc_bias.push_back(fc6_biases_data);

	file_path = base_path + "vgg_16-fc7-weights.npy";
	npy::LoadArrayFromNumpy(file_path, fc7_weights_shape, fc7_weights_data);
	file_path = base_path + "vgg_16-fc7-biases.npy";
	npy::LoadArrayFromNumpy(file_path, fc7_biases_shape, fc7_biases_data);
	fc_paras.push_back(fc7_weights_data);
	fc_bias.push_back(fc7_biases_data);

	file_path = base_path + "vgg_16-fc8-weights.npy";
	npy::LoadArrayFromNumpy(file_path, fc8_weights_shape, fc8_weights_data);
	file_path = base_path + "vgg_16-fc8-biases.npy";
	npy::LoadArrayFromNumpy(file_path, fc8_biases_shape, fc8_biases_data);
	fc_paras.push_back(fc8_weights_data);
	fc_bias.push_back(fc8_biases_data);
	// ****************************************************************************

	file_path = base_path + "vgg_16-mean_rgb.npy";
	npy::LoadArrayFromNumpy(file_path, mean_rgb_shape, mean_rgb_data);
	
	
	int argc = 4;
	const char* argv[4];
	argv[0] = "conv";
	argv[1] = "bird.JPEG";
	argv[2] = "0";
	argv[3] = "sigmoid";

	if (argc < 2) {
		std::cerr << "usage: conv <image> [gpu=0] [sigmoid=0]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
	std::cerr << "GPU: " << gpu_id << std::endl;

	bool with_sigmoid = true;
	std::cerr << "With sigmoid: " << std::boolalpha << with_sigmoid << std::endl;

	cv::Mat image = load_image(argv[1]);

	cudaSetDevice(gpu_id);

	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);

	// *************************************************************************
	// 自定义卷积层 已测试
	const float alpha = 1.0f, beta = 0.0f;
	int activate_type = 1;
	int kernel_size = 3, padding = 1, stride = 1;
	int x_height = image.rows, x_width = image.cols;
	int batch_size = 1, x_channels = 3, y_channels = 3;

	// 计算卷积后图像的维数
	int y_height = (x_height - kernel_size + 2 * padding) / stride + 1;
	int y_width = (x_width - kernel_size + 2 * padding) / stride + 1;

	size_t x_bytes = batch_size * x_channels * x_height * x_width * sizeof(float);
	float* x = nullptr;
	cudaMalloc(&x, x_bytes);
	cudaMemcpy(x, image.ptr<float>(0), x_bytes, cudaMemcpyHostToDevice);

	size_t y_bytes = batch_size * y_channels * y_height * y_width * sizeof(float);
	float* y = nullptr;
	cudaMalloc(&y, y_bytes);
	cudaMemcpy(y, 0, y_bytes, cudaMemcpyHostToDevice);

	// *************************************************************************
	vgg16_forward(x_height, x_width, x, conv_kernels, conv_bias, fc_paras, fc_bias, y);
	// *************************************************************************

	// *************************************************************************
	// clang-format off
	/*const float kernel_template[3][3] = {
		{ 1, 1, 1 },
		{ 1, -8, 1 },
		{ 1, 1, 1 }
	};*/
	const float kernel_template[3][3] = {
		{ 0, -1, -1 },
		{ 1, 0, -1 },
		{ 1, 1, 0 }
	};
	// clang-format on
	float h_kernel[3][3][3][3]; // NCHW
	for (int kernel = 0; kernel < 3; ++kernel) {
		for (int channel = 0; channel < 3; ++channel) {
			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 3; ++column) {
					h_kernel[kernel][channel][row][column] = kernel_template[row][column];
				}
			}
		}
	}
	float *d_kernel = nullptr, *d_bias = nullptr;
	cudaMalloc(&d_kernel, sizeof(h_kernel));
	cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
	// *************************************************************************
	// *************************************************************************
	conv_foward_layer(
		cudnn,
		batch_size,
		x_channels,
		x_height,
		x_width,
		x,
		kernel_size,
		d_kernel,
		d_bias,
		padding,
		stride,
		y_channels,
		y_height,
		y_width,
		y_bytes,
		y,
		activate_type);
	float* y_output1 = new float[y_bytes];
	cudaMemcpy(y_output1, y, y_bytes, cudaMemcpyDeviceToHost);
	save_image("./yy1.png", y_output1, y_height, y_width);
	/*conv_foward_layer(
		cudnn,
		&alpha,
		batch_size,
		y_channels,
		y_height,
		y_width,
		y,
		kernel_size,
		d_kernel,
		padding,
		stride,
		&beta,
		y_channels,
		y_height,
		y_width,
		y_bytes,
		y,
		activate_type);
	float* y_output = new float[y_bytes];
	cudaMemcpy(y_output, y, y_bytes, cudaMemcpyDeviceToHost);
	save_image("./yy2.png", y_output, y_height, y_width);*/
	// *************************************************************************
	// pooling 
	const int pooling_size = 8;
	int p_height = y_height / pooling_size;
	int p_width = y_width / pooling_size;
	size_t p_bytes = batch_size * y_channels * p_height * p_width * sizeof(float);
	float* p = nullptr;
	cudaMalloc(&p, p_bytes);
	cudaMemcpy(p, 0, p_bytes, cudaMemcpyHostToDevice);
	pool_forward_layer(cudnn,
		&alpha,
		batch_size,
		y_channels,
		y_height,
		y_width,
		y,
		pooling_size,
		&beta,
		p_height,
		p_width,
		p,
		1);
	float* p_output1 = new float[p_bytes];
	cudaMemcpy(p_output1, p, p_bytes, cudaMemcpyDeviceToHost);
	save_image("./yyap.png", p_output1, p_height, p_width);
	// *************************************************************************

	delete[] y_output1;
	cudaFree(d_kernel);
	cudaFree(x);
	cudaFree(y);
	// 销毁
	cudnnDestroy(cudnn);
}