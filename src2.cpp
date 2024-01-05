#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/highgui/highgui.hpp>


#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>


class HairSegmentationModel
	/*
	Class for hair segmentation
		model_dir (string): model directory (.onnx)
		use_cuda_opencv (bool): Use cuda to run the preprocessing step
		use_cuda_model (bool): use cuda to speed up model inference
	*/
{
public:

	HairSegmentationModel(std::string model_dir, bool use_cuda_opencv, bool use_cuda_model) : model_dir(model_dir), use_cuda_opencv(use_cuda_opencv), use_cuda_model(use_cuda_model) {}
	bool init_done = false;
	Ort::Env env;
	Ort::AllocatorWithDefaultOptions ort_alloc;
	Ort::RunOptions runOptions;
	//std::string model_dir = "C:\\Users\\DELL\ Inspiron\\Downloads\\fused_bisenet.onnx"; // Replace the model directory here
	std::string model_dir;

	bool use_cuda_opencv;
	bool use_cuda_model;

	std::wstring wModel_path = std::wstring(model_dir.begin(), model_dir.end());
	int model_height = 360;
	int model_width = 360;
	const std::array<int64_t, 4> inputShape = { 1, 3, model_height, model_width };                      // Define input shape
	const std::array<int64_t, 4> outputShape = { 1, 2, model_height, model_width };
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);



	Ort::Session session = Ort::Session(env, wModel_path.c_str(), set_config());

	//Ort::Session session;

	Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
	Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
	std::array<const char*, 1> inputNames = { inputName.get() };
	std::array<const char*, 1> outputNames = { outputName.get() };



	Ort::SessionOptions set_config()
	{
		Ort::SessionOptions sessionOptions;
		sessionOptions.SetIntraOpNumThreads(1);

		//// The 2 lines below activate gpu
		if (use_cuda_model)
		{
			OrtCUDAProviderOptions cuda_options{};
			sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
		}

		sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

		return sessionOptions;
	}

	cv::Mat preprocess_image(cv::Mat imageBGR, int sizeX = 360, int sizeY = 360)
	{
		if (use_cuda_opencv)
		{
			// Implement later
			//cv::cuda::GpuMat image_gpu;
			//cv::Mat processed_image;
			//image_gpu.upload(image_bgr);
			//cv::cuda::resize(image_gpu, image_gpu, { sizeX, sizeY }, cv::InterpolationFlags::INTER_CUBIC);
			//cv::cuda::cvtColor(image_gpu, image_gpu, cv::ColorConversionCodes::COLOR_BGR2RGB);
			//image_gpu.convertTo(image_gpu, CV_32F, 1.0 / 255);

			//image_gpu.download(processed_image);
			//cv::dnn::blobFromImage(processed_image, processed_image);

		}


		cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
		cv::resize(imageBGR, resizedImageBGR, { sizeX, sizeY }, cv::InterpolationFlags::INTER_CUBIC);
		cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
		resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
		cv::dnn::blobFromImage(resizedImage, preprocessedImage);


		return preprocessedImage;
	}


	cv::Mat execute(cv::Mat image)
	{
		constexpr int64_t numChannels = 3;
		constexpr int64_t width = 360;
		constexpr int64_t height = 360;
		constexpr int64_t numClasses = 2;
		constexpr int64_t numInputElements = numChannels * height * width;
		constexpr int64_t numOutputElements = 1 * numClasses * height * width;

		std::vector<float> input(numInputElements);                                                       // Define the type of the input
		std::vector<float> results(numOutputElements);

		auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());          // Create a tensor for the input
		auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());   // Create a tensor

		cv::Mat imageVec = preprocess_image(image);
		std::copy(imageVec.begin<float>(), imageVec.end<float>(), input.begin());

		try {
			session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
		}
		catch (Ort::Exception& e) {
			std::cout << e.what() << std::endl;
			return cv::Mat();
		}

		cv::Mat mask(360, 360, CV_32FC1, results.data());
		cv::Mat resultImage;
		cv::threshold(mask, resultImage, 0, 1, cv::THRESH_BINARY);
		cv::resize(resultImage, resultImage, { image.cols, image.rows }, 0, 0, cv::INTER_NEAREST);
		return resultImage;

	}
};



cv::Mat load_image(std::string img_dir, int sizeX = 360, int sizeY = 360)
{
	cv::Mat imageBGR = cv::imread(img_dir);
	cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
	cv::resize(imageBGR, resizedImageBGR, { sizeX, sizeY }, cv::InterpolationFlags::INTER_CUBIC);
	cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
	resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
	cv::dnn::blobFromImage(resizedImage, preprocessedImage);
	return preprocessedImage;
}

int main()
{
	//bool use_cuda = false;
	std::string img_dir = "C:/Users/BinhDS/Downloads/image.jpg";

	//// ++++++++++++++++++++++++++++++++++++++++++++  IMAGE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//// ++++++++++++++++++++++++++++++++++++++++++++  MODEL  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	HairSegmentationModel bisenet("C:/Users/BinhDS/Downloads/fused_bisenet.onnx", true, false);
	//cv::Mat image = cv::imread(img_dir);
	//cv::Mat result = bisenet.execute(image);


	cv::Mat frame;
	cv::Mat result;
	cv::VideoCapture video("C:/Users/BinhDS/Downloads/timi.mp4");

	if (!video.isOpened())
	{
		std::cout << "Failed to open the video file." << std::endl;
		return -1;
	}
	cv::namedWindow("video", cv::WINDOW_NORMAL);
	float mean_fps = 0.0;
	int cnt = 0;

	// cpu: 433.435
	// gpu: 204.934

	// cpu: 12.97
	// gpu: 100
	while (true)
	{

		if (!video.read(frame)) { break; }

		cv::imshow("Video", frame);
		if (cv::waitKey(25) == 'q') { break; };
	}
	video.release();
	cv::destroyAllWindows();
	//std::cout << result.size() << std::endl;


	return 0;
}
