#include "net.h"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

// ����ؼ���ṹ�壺�㣬����
struct KeyPoint {
	cv::Point2f p;
	float prob;
};

// ��������
static int detect_posenet(const cv::Mat& bgr, std::vector<KeyPoint>& keypoints) {
	ncnn::Net posenet;
	// ʹ��bf16��������bf16���õ��ڴ�ռ��fp32��һ��
	posenet.opt.use_packing_layout = true;
	posenet.opt.use_bf16_storage = true;
	// ����ģ��
	posenet.load_param("C:/Users/Admin/Desktop/project_cpp/Project1/pose.param");
	posenet.load_model("C:/Users/Admin/Desktop/project_cpp/Project1/pose.bin");
	// ��ȡ���
	int w = bgr.cols;
	int h = bgr.rows;
	// from_pixels_resize������opencv��ʽ��ͼƬתΪncnn��ʽ��ͼƬ�����������ǰ������
	// from_pixels_resize���������ͼƬ��������
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 192, 256);
	// �����ֵ�ͷ���
	const float mean_vals[3] = { 0.458f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
	const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.244f / 255.f, 1 / 0.225f / 255.f };
	// ��������м���ֵ������Ĳ���
	in.substract_mean_normalize(mean_vals, norm_vals);
	// ncnn����: ģ������ ===> �½�һ��Extractor ===> �������� ===> ��ȡ���
	// �½�Extractor
	ncnn::Extractor ex = posenet.create_extractor();
	// ��������
	ex.input("data", in);
	ncnn::Mat out;
	// ��ȡ���
	ex.extract("conv3_fwd", out);

	keypoints.clear();
	// ��ȡģ�͵Ķ���������뵽KeyPoint֮��
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
		// ���ƹؼ���֮�������
		cv::line(image, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
	}

	// draw joint
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		const KeyPoint& keypoint = keypoints[i];

		fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);
		// С��0.2��ֵ�Ľ������Ҫ
		if (keypoint.prob < 0.2f)
			continue;
		// ���ƹؼ���
		cv::circle(image, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
	}
	cv::imwrite("./pose_check.jpg", image);
	cv::imshow("ԭͼ", image);
	cv::waitKey(50000);

}

int main()
{
	std::string imagepath = "C:/Users/Admin/Desktop/project_cpp/Project1/test.jpg";  // ͼ��·��

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
