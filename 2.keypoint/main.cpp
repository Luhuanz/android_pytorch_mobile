#include "net.h"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

// 定义关键点结构体：点，概率
struct KeyPoint {
	cv::Point2f p;
	float prob;
};

// 推理网络
static int detect_posenet(const cv::Mat& bgr, std::vector<KeyPoint>& keypoints) {
	ncnn::Net posenet;
	// 使用bf16进行推理，bf16所用的内存空间比fp32少一半
	posenet.opt.use_packing_layout = true;
	posenet.opt.use_bf16_storage = true;
	// 加载模型
	posenet.load_param("C:/Users/Admin/Desktop/project_cpp/Project1/pose.param");
	posenet.load_model("C:/Users/Admin/Desktop/project_cpp/Project1/pose.bin");
	// 获取宽高
	int w = bgr.cols;
	int h = bgr.rows;
	// from_pixels_resize函数将opencv格式的图片转为ncnn格式的图片，用于网络的前向推理
	// from_pixels_resize函数还会对图片进行缩放
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 192, 256);
	// 定义均值和方差
	const float mean_vals[3] = { 0.458f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
	const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.244f / 255.f, 1 / 0.225f / 255.f };
	// 对输入进行减均值除方差的操作
	in.substract_mean_normalize(mean_vals, norm_vals);
	// ncnn流程: 模型载入 ===> 新建一个Extractor ===> 设置输入 ===> 获取输出
	// 新建Extractor
	ncnn::Extractor ex = posenet.create_extractor();
	// 设置输入
	ex.input("data", in);
	ncnn::Mat out;
	// 获取输出
	ex.extract("conv3_fwd", out);

	keypoints.clear();
	// 获取模型的而输出并填入到KeyPoint之中
	for (int p = 0; p < out.c; p++) {
		const ncnn::Mat m = out.channel(p);
		float max_prob = 0.f;
		int max_x = 0;
		int max_y = 0;
		for (int y = 0; y < out.h; y++) {
			const float* ptr = m.row(y);
			for (int x = 0; x < out.w; x++) {
				float prob = ptr[x];
				if (prob > max_prob) {
					max_prob = prob;
					max_x = x;
					max_y = y;
				}
			}
		}
		KeyPoint keypoint;
		keypoint.p = cv::Point2f(max_x*w / (float)out.w, max_y * h / (float)out.h);
		keypoint.prob = max_prob;

		keypoints.push_back(keypoint);
	}
	return 0;
}

static void draw_pose(const cv::Mat& bgr, const std::vector<KeyPoint>& keypoints)
{
	cv::Mat image = bgr.clone();

	// draw bone
	static const int joint_pairs[16][2] = {
		{0, 1}, {1, 3}, {0, 2}, {2, 4}, 
		{5, 6}, {5, 7}, {7, 9}, {6, 8}, 
		{8, 10}, {5, 11}, {6, 12}, {11, 12}, 
		{11, 13}, {12, 14}, {13, 15}, {14, 16}
	};

	for (int i = 0; i < 16; i++)
	{
		const KeyPoint& p1 = keypoints[joint_pairs[i][0]];
		const KeyPoint& p2 = keypoints[joint_pairs[i][1]];

		if (p1.prob < 0.2f || p2.prob < 0.2f)
			continue;
		// 绘制关键点之间的连线
		cv::line(image, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
	}

	// draw joint
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		const KeyPoint& keypoint = keypoints[i];

		fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);
		// 小于0.2阈值的结果不需要
		if (keypoint.prob < 0.2f)
			continue;
		// 绘制关键点
		cv::circle(image, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
	}
	cv::imwrite("./pose_check.jpg", image);
	cv::imshow("原图", image);
	cv::waitKey(50000);

}

int main()
{
	std::string imagepath = "C:/Users/Admin/Desktop/project_cpp/Project1/test.jpg";  // 图像路径

	cv::Mat m = cv::imread(imagepath, 1);
	if (m.empty())
	{
		fprintf(stderr, "%s\n", imagepath.c_str());
		return -1;
	}

	std::vector<KeyPoint> keypoints;
	detect_posenet(m, keypoints);
	draw_pose(m, keypoints);

	return 0;
}
