#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;

class HairSegmentationModel {

public:
	Ort::Env env;
	Ort::AllocatorWithDefaultOptions ort_alloc;
	Ort::RunOptions runOptions;
	std::string model_dir = "C:\\Users\\DELL\ Inspiron\\Downloads\\fused_bisenet.onnx"; // Replace the model directory here
	std::wstring wModel_path = std::wstring(model_dir.begin(), model_dir.end());
	int model_height = 360;
	int model_width = 360;
	const std::array<int64_t, 4> inputShape = { 1, 3, model_height, model_width };                      // Define input shape
	const std::array<int64_t, 4> outputShape = {1, 2, model_height, model_width};
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	Ort::Session session = Ort::Session(env, wModel_path.c_str(), Ort::SessionOptions{ nullptr });
	

	Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
	Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
	std::array<const char*, 1> inputNames = { inputName.get() };
	std::array<const char*, 1> outputNames = { outputName.get() };
	

	static std::vector<float> preprocess_image(cv::Mat image, int sizeX = 360, int sizeY = 360)
	{
		cv::Mat processedImage = image.clone();  // Create a copy of the input image

		cv::cvtColor(processedImage, processedImage, cv::COLOR_BGR2RGB);
		cv::resize(processedImage, processedImage, { sizeX, sizeY }, 0, 0, cv::INTER_NEAREST);
		processedImage = processedImage.reshape(1, 1);
		std::vector<float> vec;
		processedImage.convertTo(vec, CV_32FC1, 1.0 / 255);

		std::vector<float> output;
		for (size_t ch = 0; ch < 3; ++ch)
		{
			for (size_t i = ch; i < vec.size(); i += 3)
			{
				output.emplace_back(vec[i]);
			}
		}
		return output;
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

		std::vector<float> imageVec = preprocess_image(image);
		std::copy(imageVec.begin(), imageVec.end(), input.begin());

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


int main()
{
	std::string model_dir = "C:/Users/DELL\ Inspiron/Downloads/further_fused_bisenet.onnx";
	std::string image_path = "C:/Users/DELL\ Inspiron/Downloads/dog.png";
	HairSegmentationModel bisenet_model;

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
