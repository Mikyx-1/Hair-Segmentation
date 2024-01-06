#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


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
	Ort::Env env;
	Ort::AllocatorWithDefaultOptions ort_alloc;
	Ort::RunOptions runOptions;
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

	// Config size for the model
	 int64_t numChannels = 3;
	 int64_t width = 360;
	 int64_t height = 360;
	 int64_t numClasses = 2;
	 int64_t numInputElements = numChannels * height * width;
	 int64_t numOutputElements = 1 * numClasses * height * width;


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
		return 1-resultImage;

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

	HairSegmentationModel bisenet_model("C:/Users/DELL\ Inspiron/Downloads/fused_bisenet.onnx", false, false);


	cv::Mat frame;
	cv::Mat result;

	cv::VideoCapture video("C:/Users/DELL\ Inspiron/Downloads/timi.mp4");
	if (!video.isOpened())
	{
		std::cout << "Failed to open the video file." << std::endl;
		return -1;
	}
	cv::namedWindow("video", cv::WINDOW_NORMAL);
	while (true)
	{
		cv::Mat frame;
		if (!video.read(frame)) { break; }
		cv::Mat result = bisenet_model.execute(frame);

		cv::imshow("Video", result);
		
		if (cv::waitKey(25) == 'q') { break; };
	}
	video.release();
	cv::destroyAllWindows();
	return 0;
}
