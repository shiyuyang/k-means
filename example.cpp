#include "k-means.h"
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace cv;

void rgb2hsv(double R, double G, double B, double& H, double& S, double& V) {
	R = R / 255.0;
	G = G / 255.0;
	B = B / 255.0;
	double max_value = std::max(B,std::max(R, G));
	double min_value = std::min(B,std::min(R, G));
	double diff = max_value - min_value;
	if (max_value == min_value) {
		H = 0;
	}
	else if (max_value == R&&G >= B) {
		H = 60 * ((G - B) / diff) + 0;
	}
	else if (max_value == R&&G < B) {
		H = 60 * ((G - B) / diff) + 360;
	}
	else if (max_value == G) {
		H = 60 * ((B - R) / diff) + 120;
	}
	else if (max_value == B) {
		H = 60 * ((R - G) / diff) + 240;
	}
	if (max_value == 0) {
		S = 0;
	}
	else {
		S = diff / max_value;
	}
	V = max_value;
}

int main(int argc, char** argv) {
	auto image = imread("d:\\test.jpg");
	std::vector<std::array<double, 3>> dataset;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			if (rand() % 3 != 1) {
				continue;
			}
			std::array<double, 3> node;
			rgb2hsv(image.at<Vec3b>(i, j)[2], image.at<Vec3b>(i, j)[1], image.at<Vec3b>(i, j)[0], node[0], node[1], node[2]);
			dataset.push_back(node);
		}
	}
	leopard::kmeans<3, 20>::kmeans_init();
	leopard::kmeans<3, 20> k_means;
	auto time_now = std::chrono::system_clock::now();
	auto duration_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_now.time_since_epoch());
	std::array<leopard::cluster<3>, 20> result = k_means.get_kmeans(dataset, 0.1);
	auto time_now2 = std::chrono::system_clock::now();
	auto duration_in_ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(time_now2.time_since_epoch());
	std::cout << duration_in_ms2.count()-duration_in_ms.count() << std::endl;
	return 0;
}
