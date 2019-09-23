#include "vgg16.h"

void checkCUDNN(cudnnStatus_t status)
{
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "Error on line " << __LINE__ << ": "
			<< cudnnGetErrorString(status) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

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
	int activate_type)
{
	// 输入张量的描述
	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,	// 注意是 NHWC，TensorFlow更喜欢以 NHWC 格式存储张量(通道是变化最频繁的地方，即 BGR)，而其他一些更喜欢将通道放在前面
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batch_size,
		/*channels=*/x_channels,
		/*image_height=*/x_height,
		/*image_width=*/x_width));

	// 卷积核的描述（形状、格式）
	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,	// 注意是 NCHW
		///*format=*/CUDNN_TENSOR_NHWC,	// 注意是 NHWC ?????????????
		/*out_channels=*/y_channels,
		/*in_channels=*/x_channels,
		/*kernel_height=*/kernel_size,
		/*kernel_width=*/kernel_size));

	// 卷积操作的描述（步长、填充等等）
	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
		/*pad_height=*/padding,
		/*pad_width=*/padding,
		/*vertical_stride=*/stride,
		/*horizontal_stride=*/stride,
		/*dilation_height=*/1,
		/*dilation_width=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION, // CUDNN_CONVOLUTION
		/*computeType=*/CUDNN_DATA_FLOAT));

	// 输出张量的描述
	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batch_size,
		/*channels=*/y_channels,
		/*image_height=*/y_height,
		/*image_width=*/y_width));

	// 卷积算法的描述
	// cudnn_tion_fwd_algo_gemm——将卷积建模为显式矩阵乘法，
	// cudnn_tion_fwd_algo_fft——它使用快速傅立叶变换(FFT)进行卷积或
	// cudnn_tion_fwd_algo_winograd——它使用Winograd算法执行卷积。
	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	checkCUDNN(
		cudnnGetConvolutionForwardAlgorithm(handle,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, // CUDNN_CONVOLUTION_FWD_SPECIFY_​WORKSPACE_LIMIT（在内存受限的情况下，memoryLimitInBytes 设置非 0 值）
			/*memoryLimitInBytes=*/0,
			&convolution_algorithm));

	// 计算 cuDNN 它的操作需要多少内存
	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
		input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		convolution_algorithm,
		&workspace_bytes));
	if (workspace_bytes == 0) workspace_bytes = 8 * (size_t)1048576;
	std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
		<< std::endl;
	assert(workspace_bytes > 0);
	// 分配内存， 从 cudnnGetConvolutionForwardWorkspaceSize 计算而得
	void* y_workspace = nullptr;
	cudaMalloc(&y_workspace, workspace_bytes);

	// 从 cudnnGetConvolution2dForwardOutputDim 计算而得
	//size_t y_bytes = batch_size * y_channels * y_height * y_width * sizeof(float);
	//cudaMalloc(&y, y_bytes);
	//cudaMemset(y, 0, y_bytes);

	// 真正的卷积操作 ！！！前向卷积
	const float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnConvolutionForward(handle,
		&alpha,
		input_descriptor,
		x,
		kernel_descriptor,
		kernel,
		convolution_descriptor,
		convolution_algorithm,
		y_workspace, // 注意，如果我们选择不需要额外内存的卷积算法，d_workspace可以为nullptr。
		y_bytes,
		&beta,
		output_descriptor,
		y));

	// add conv bias 
	if (bias != nullptr)
	{
		cudnnTensorDescriptor_t bias_descriptor;
		checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
		checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
			/*format=*/CUDNN_TENSOR_NHWC,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/y_channels,
			/*bias_height=*/1,
			/*bias_width=*/1));
		checkCUDNN(cudnnAddTensor(
			handle,
			&alpha,
			bias_descriptor,
			bias,
			&beta,
			output_descriptor,
			y));
		cudnnDestroyTensorDescriptor(bias_descriptor);
	}
	
	// activate layer
	if (activate_type != -1) {

		// 描述激活
		cudnnActivationDescriptor_t activation_descriptor;
		checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
		switch (activate_type)
		{
		case CUDNN_ACTIVATION_SIGMOID:
			checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
				CUDNN_ACTIVATION_SIGMOID,
				CUDNN_PROPAGATE_NAN,
				/*relu_coef=*/0));
			break;
		case CUDNN_ACTIVATION_RELU:
			checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
				CUDNN_ACTIVATION_RELU,
				CUDNN_PROPAGATE_NAN,
				/*relu_coef=*/0));
			break;
		default:
			break;
		}
		
		// 前向 sigmoid 激活函数
		checkCUDNN(cudnnActivationForward(handle,
			activation_descriptor,
			&alpha,
			output_descriptor,
			y,
			&beta,
			output_descriptor,
			y));
		cudnnDestroyActivationDescriptor(activation_descriptor);
	}
	/*if (x_channels == y_channels && y_height == x_height)
	{
		cudnnAddTensor(
			handle,
			&alpha,
			input_descriptor,
			x,
			&beta,
			output_descriptor,
			y);
	}*/
	/*float* y_output = new float[y_bytes];
	cudaMemcpy(y_output, y, y_bytes, cudaMemcpyDeviceToHost);
	save_image("./cv.png", y_output, y_height, y_width);*/

	cudaFree(y_workspace);
	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}

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
	int pool_type)
{

	// 输入张量的描述
	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,	// 注意是 NHWC，TensorFlow更喜欢以 NHWC 格式存储张量(通道是变化最频繁的地方，即 BGR)，而其他一些更喜欢将通道放在前面
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batch_size,
		/*channels=*/x_channels,
		/*image_height=*/x_height,
		/*image_width=*/x_width));

	// 输出张量的描述
	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batch_size,
		/*channels=*/x_channels,
		/*image_height=*/y_height,
		/*image_width=*/y_width));

	// pooling的描述
	cudnnPoolingDescriptor_t pooling_descriptor;
	checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
	switch (pool_type)
	{
	case 0:
		checkCUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor,
			CUDNN_POOLING_MAX,
			CUDNN_PROPAGATE_NAN,
			pooling_size,
			pooling_size,
			0,
			0,
			pooling_size,
			pooling_size));
		break;
	case 1:
		checkCUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor,
			CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
			CUDNN_PROPAGATE_NAN,
			pooling_size,
			pooling_size,
			0,
			0,
			pooling_size,
			pooling_size));
		break;
	default:
		break;
	}

	// pooling计算
	cudnnPoolingForward(
		handle,
		pooling_descriptor,
		alpha,
		input_descriptor,
		x,
		beta,
		output_descriptor,
		y);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyPoolingDescriptor(pooling_descriptor);
}

void vgg16_forward(
	const int& x_height,
	const int& x_width,
	const void *x,
	std::vector<std::vector<float>> conv_kernels,
	std::vector<std::vector<float>> conv_bias,
	std::vector<std::vector<float>> fc_kernels,
	std::vector<std::vector<float>> fc_bias,
	void *y)
{
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
	const float alpha = 1.0f, beta = 0.0f;

	int in_height = x_height, in_width = x_width;
	int out_height = 0, out_width = 0;

	size_t input_bytes = batch_size * conv_kernel_sizes[0][0] * in_height * in_width * sizeof(float);
	float* input_tensor = nullptr;
	cudaMalloc(&input_tensor, input_bytes);
	cudaMemcpy(input_tensor, x, input_bytes, cudaMemcpyHostToDevice);

	float* output_tensor = nullptr;
	size_t out_bytes = 0;

	size_t p_bytes = 0; // pooling size

	// calculate convolution foward layers
	for (int layer = 0; layer < 13; layer++)
	{
		// 计算输出张量的维度及内存大小
		out_height = (in_height - conv_kernel_sizes[layer][1] + 2 * padding) / stride + 1;
		out_width = (in_width - conv_kernel_sizes[layer][1] + 2 * padding) / stride + 1;
		out_bytes = batch_size * conv_kernel_sizes[layer][2] * out_height * out_width * sizeof(float);
		cudaMalloc(&output_tensor, out_bytes);
		cudaMemcpy(output_tensor, 0, out_bytes, cudaMemcpyHostToDevice);

		// *************************************************************************
		// load conv kernel
		float* d_kernel = nullptr;
		cudaMalloc(&d_kernel, sizeof(float)*conv_kernels[layer].size());
		cudaMemcpy(d_kernel, conv_kernels[layer].data(), sizeof(float)*conv_kernels[layer].size(), cudaMemcpyHostToDevice);
		//load conv bias
		float* d_bias = nullptr;
		cudaMalloc(&d_bias, sizeof(float)*conv_bias[layer].size());
		cudaMemcpy(d_bias, conv_bias[layer].data(), sizeof(float)*conv_bias[layer].size(), cudaMemcpyHostToDevice);
		// *************************************************************************
		
		// *************************************************************************
		conv_foward_layer(
			cudnn,
			batch_size,
			/*x_channels*/ conv_kernel_sizes[layer][0],
			in_height,
			in_width,
			input_tensor,
			/*kernel_size*/ conv_kernel_sizes[layer][1],
			d_kernel,
			d_bias,
			padding,
			stride,
			/*y_channels*/ conv_kernel_sizes[layer][2],
			out_height,
			out_width,
			out_bytes,
			output_tensor,
			1);

		//delete [] h_kernel;
		// *************************************************************************

		// *************************************************************************
		if (conv_kernel_sizes[layer][3] == 1)
		{
			int p_height = out_height / pooling_size;
			int p_width = out_width / pooling_size;
			p_bytes = batch_size * conv_kernel_sizes[layer][2] * p_height * p_width * sizeof(float);
			
			float* p = nullptr;
			cudaMalloc(&p, p_bytes);
			cudaMemcpy(p, 0, p_bytes, cudaMemcpyHostToDevice);

			pool_forward_layer(cudnn,
				&alpha,
				batch_size,
				conv_kernel_sizes[layer][2],
				out_height,
				out_width,
				output_tensor,
				pooling_size,
				&beta,
				p_height,
				p_width,
				p,
				0);

			out_height = p_height;
			out_width = p_width;
			out_bytes = p_bytes;
			cudaFree(output_tensor);
			output_tensor = nullptr;
			cudaMalloc(&output_tensor, out_bytes);
			cudaMemcpy(output_tensor, p, out_bytes, cudaMemcpyHostToDevice);
			
			cudaFree(p);
		}

		in_height = out_height;
		in_width = out_width;
		input_bytes = out_bytes;

		cudaFree(input_tensor);
		input_tensor = nullptr;
		cudaMalloc(&input_tensor, input_bytes);
		cudaMemcpy(input_tensor, output_tensor, input_bytes, cudaMemcpyHostToDevice);

		float* y_output1 = new float[input_bytes];
		cudaMemcpy(y_output1, input_tensor, input_bytes, cudaMemcpyDeviceToHost);
		save_image("./layer"+ std::to_string(layer+1)+".png", y_output1, in_height, in_width);

		cudaFree(output_tensor);
		output_tensor = nullptr;
		cudaFree(d_kernel);
		d_kernel = nullptr;
		cudaFree(d_bias);
		d_bias = nullptr;


		//int _n, _c, _h, _w, _tmp;
		//cudnnDataType_t _t;
		//checkCUDNN(cudnnGetTensor4dDescriptor(input_tensor, &_t, &_n, &_c, &_h, &_w, &_tmp, &_tmp, &_tmp, &_tmp));
		//std::cout << "batch: " << _n               /* number of inputs (batch size) */
		//	<< " channel: " << _c                   /* number of input feature maps  */
		//	<< " height: " << _h                   /* height of input section */
		//	<< " width: " << _w                   /* width of input section */
		//	<< " \n" << std::endl;
		std::cout << "batch: " << sizeof(input_tensor) * 1048576 / (in_height * in_width * conv_kernel_sizes[layer][2])
			<< " channel: " << sizeof(input_tensor) * 1048576 / (in_height * in_width * batch_size)
			<< " height: " << sizeof(input_tensor) * 1048576 / (conv_kernel_sizes[layer][2] * in_width * batch_size)
			<< " width: " << sizeof(input_tensor) * 1048576 / (conv_kernel_sizes[layer][2] * in_height * batch_size)
			<< "\n" << std::endl;
	}

	// calculate fully connected foward layers
	for (int fc_layer = 0; fc_layer < 3; fc_layer++)
	{
		// 计算输出张量的维度及内存大小
		out_height = 1;
		out_width = 1;
		out_bytes = batch_size * fc_sizes[fc_layer] * out_height * out_width * sizeof(float);
		cudaMalloc(&output_tensor, out_bytes);
		cudaMemcpy(output_tensor, 0, out_bytes, cudaMemcpyHostToDevice);

		// *************************************************************************
		// load fully connected paras
		float* d_fc_kernel = nullptr;
		cudaMalloc(&d_fc_kernel, sizeof(float) * fc_kernels[fc_layer].size() * in_height * in_width);
		cudaMemcpy(d_fc_kernel, fc_kernels[fc_layer].data(), sizeof(float) * fc_kernels[fc_layer].size() * in_height * in_width, cudaMemcpyHostToDevice);
		//load fully connected bias
		float* d_fc_bias = nullptr;
		cudaMalloc(&d_fc_bias, sizeof(float)*fc_bias[fc_layer].size());
		cudaMemcpy(d_fc_bias, fc_bias[fc_layer].data(), sizeof(float) * fc_bias[fc_layer].size(), cudaMemcpyHostToDevice);
		// *************************************************************************

		// *************************************************************************
		int x_channels = 0, fc_kernel_size = 0;
		if (fc_layer == 0)
		{
			x_channels = conv_kernel_sizes[12][2];
			fc_kernel_size = in_height;
		}
		else
		{
			x_channels = fc_sizes[fc_layer - 1];
			fc_kernel_size = 1;
		}
		// perform fully connected operation
		conv_foward_layer(
			cudnn,
			batch_size,
			/*x_channels*/x_channels,
			in_height,
			in_width,
			input_tensor,
			/*kernel_size*/fc_kernel_size,
			d_fc_kernel,
			d_fc_bias,
			0,
			1,
			/*y_channels*/fc_sizes[fc_layer],
			out_height,
			out_width,
			out_bytes,
			output_tensor,
			1);

		in_height = out_height;
		in_width = out_width;
		input_bytes = out_bytes;
		cudaFree(input_tensor);
		input_tensor = nullptr;
		cudaMalloc(&input_tensor, input_bytes);
		cudaMemcpy(input_tensor, output_tensor, input_bytes, cudaMemcpyHostToDevice);

		cudaFree(output_tensor);
		output_tensor = nullptr;
		cudaFree(d_fc_kernel);
		d_fc_kernel = nullptr;
		cudaFree(d_fc_bias);
		d_fc_bias = nullptr;
	}

	cudaMemcpy(y, input_tensor, input_bytes, cudaMemcpyHostToDevice);

	cudaFree(output_tensor);
	cudnnDestroy(cudnn);
}