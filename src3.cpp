#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

cv::Mat loadImage(std::string imageFilepath, int sizeX = 224, int sizeY = 224)
{
	cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
	cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;

	cv::resize(imageBGR, resizedImageBGR, { sizeX, sizeY }, cv::InterpolationFlags::INTER_CUBIC);
	cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
	resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

	cv::dnn::blobFromImage(resizedImage, preprocessedImage);
	return preprocessedImage;
}

cv::Mat processImage(cv::Mat imageBGR, int sizeX = 224, int sizeY = 224)
{
	cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
	cv::resize(imageBGR, resizedImageBGR, { sizeX, sizeY }, cv::InterpolationFlags::INTER_CUBIC);
	cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
	resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
	cv::dnn::blobFromImage(resizedImage, preprocessedImage);
	return preprocessedImage;

}

int main()
{
	bool use_cuda = false;
	std::string img_dir = "C:/Users/BinhDS/Downloads/image.jpg";


	// ++++++++++++++++++++++++++++++++++++++++++++  IMAGE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	cv::Mat blob = loadImage(img_dir);
	// ++++++++++++++++++++++++++++++++++++++++++++  MODEL  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	std::string instanceName{ "image-classification-inference" };
	std::string modelFilepath = "C:/Users/BinhDS/Downloads/resnet18.onnx";
	std::wstring w_path = std::wstring(modelFilepath.begin(), modelFilepath.end());

	const std::array<int64_t, 4> inputShape = { 1, 3, 224, 224 };                      // Define input shape
	const std::array<int64_t, 2> outputShape = { 1, 1000 };

	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
		instanceName.c_str());
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);

	// The 2 lines below activate gpu
	OrtCUDAProviderOptions cuda_options{};
	sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	Ort::Session session(env, w_path.c_str(), sessionOptions);

	Ort::AllocatorWithDefaultOptions allocator;
	size_t numInputNodes = session.GetInputCount();
	size_t numOutputNodes = session.GetOutputCount();

	Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, allocator);
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);

	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

	std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

	Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, allocator);
	Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
	std::vector<int64_t> outputDims = outputTensorInfo.GetShape();


	std::array<const char*, 1> inputNames = { inputName.get() };
	std::array<const char*, 1> outputNames = { outputName.get() };

	std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;


	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

	constexpr int64_t numInputElements = 3 * 224 * 224;
	constexpr int64_t numOutputElements = 1 * 1000;
	std::vector<float> input(numInputElements);
	std::vector<float> results(numOutputElements);

	auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, input.data(), input.size(), inputShape.data(), inputShape.size());          // Create a tensor for the input
	auto outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, results.data(), results.size(), outputShape.data(), outputShape.size());   // Create a tensor

	std::copy(blob.begin<float>(), blob.begin<float>(), input.begin());


	for (int i = 0; i < 100000; i++)
	{
		std::cout << "Iteration: " << i << std::endl;
		try {
			session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
		}
		catch (Ort::Exception& e) {
			std::cout << e.what() << std::endl;
		}
		//cv::Mat blob = loadImage(img_dir);

	}


	//for (size_t i = 0; i < results.size(); ++i) {
	//	std::cout << results[i] << std::endl;
	//}

	std::cout << "Done!" << std::endl;


	return 0;
}

cv::Mat process_image_gpu(cv::Mat image_bgr, int sizeX = 360, int sizeY = 360)
{
	cv::cuda::GpuMat image_gpu;
	cv::Mat processed_image;
	image_gpu.upload(image_bgr);
	cv::cuda::resize(image_gpu, image_gpu, { sizeX, sizeY }, cv::InterpolationFlags::INTER_CUBIC);
	cv::cuda::cvtColor(image_gpu, image_gpu, cv::ColorConversionCodes::COLOR_BGR2RGB);
	image_gpu.convertTo(image_gpu, CV_32F, 1.0 / 255);

	image_gpu.download(processed_image);
	cv::dnn::blobFromImage(processed_image, processed_image);
	return processed_image;
}
