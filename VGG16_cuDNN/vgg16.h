#ifndef _VGG16_H_
#define _VGG16_H_

#include <cudnn.h>
#include <iostream>
#include <assert.h>
#include "util.h"
#include <vector>

const int batch_size = 1, pooling_size = 2;
const int padding = 1, stride = 1;
const int conv_blocks[5] = { 2, 2, 3, 3, 3 };
const int conv_kernel_sizes[13][4] = {
	// block1 1x224x224x3 NHWC
	{ 3, 3, 64, 0 }, // in_c, k_size, out_c, need_pooling
	{ 64, 3, 64, 1 }, 

	// block2
	{ 64, 3, 128, 0 }, 
	{ 128, 3, 128, 1 },

	// block3
	{ 128, 3, 256, 0 },
	{ 256, 3, 256, 0 },
	{ 256, 3, 256, 1 },

	// block4
	{ 256, 3, 512, 0 },
	{ 512, 3, 512, 0 },
	{ 512, 3, 512, 1 },

	// block5
	{ 512, 3, 512, 0 },
	{ 512, 3, 512, 0 },
	//{ 512, 3, 512, 1 },
	{ 512, 3, 512, 1 }
	
};
//const int conv_kernel_sizes[13][4] = {
//	// block1 1x224x224x3 NHWC
//	{ 3, 3, 3, 0 }, // in_c, k_size, out_c, need_pooling
//	{ 3, 3, 3, 1 },
//
//	// block2
//	{ 3, 3, 3, 0 },
//	{ 3, 3, 3, 1 },
//
//	// block3
//	{ 3, 3, 3, 0 },
//	{ 3, 3, 3, 0 },
//	{ 3, 3, 3, 1 },
//
//	// block4
//	{ 3, 3, 3, 0 },
//	{ 3, 3, 3, 0 },
//	{ 3, 3, 3, 1 },
//
//	// block5
//	{ 3, 3, 3, 0 },
//	{ 3, 3, 3, 0 },
//	//{ 512, 3, 512, 1 },
//	{ 3, 3, 3, 1 }
//};
const int fc_sizes[3] = {
	4096,
	4096,
	1000
};

void checkCUDNN(cudnnStatus_t status);

/**
*  前向卷积传播层，
*  @param cudnn
*  @param input_descriptor: CUDNN_TENSOR_NHWC
*  @param kernel_descriptor
*  @param convolution_descriptor
*  @param output_descriptor: CUDNN_TENSOR_NHWC
*  @param convolution_algorithm
*
*  @return void
*/
void conv_foward_layer(
	cudnnHandle_t& handle,
	const int& batch_size,
	const int& x_channels,
	const int& x_height,
	const int& x_width,
	const void *x,
	const int& kernel_size,
	const void *kernel,
	const void *bias,
	const int& padding,
	const int& stride,
	const int& y_channels,
	const int& y_height,
	const int& y_width,
	const size_t& y_bytes,
	void *y,
	int activate_type);

void pool_forward_layer(cudnnHandle_t& handle,
	const void *alpha,
	const int& batch_size,
	const int& x_channels,
	const int& x_height,
	const int& x_width,
	const void *x,
	const int& pooling_size,
	const void *beta,
	const int& y_height,
	const int& y_width,
	void *y,
	int pool_type);

void vgg16_forward(
	const int& x_height,
	const int& x_width,
	const void *x,
	std::vector<std::vector<float>> conv_kernels,
	std::vector<std::vector<float>> conv_bias,
	std::vector<std::vector<float>> fc_kernels,
	std::vector<std::vector<float>> fc_bias,
	void *y);

#endif // !_VGG16_H_