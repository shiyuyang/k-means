#ifndef K_MEANS_H
#define K_MEANS_H
#include <vector>
#include <array>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cassert>

namespace leopard {
	template <size_t DIM>
	struct cluster {
		cluster() {
			count = 0;
			node.fill(0);
		}
		int count;
		std::array<double, DIM> node;
	};

	template <size_t DIM, size_t K>
	class kmeans {
	public:
		static void kmeans_init();
		std::array<cluster<DIM>, K>& get_kmeans(std::vector<std::array<double, DIM>>& dataset, double threshold);
	private:
		size_t get_centroid(std::array<double, DIM> & node, std::array<cluster<DIM>, K>& centroids);
		double get_distance(std::array<double, DIM> & node_left, std::array<double, DIM> & node_right);
		void init_centroids(std::vector<std::array<double, DIM>>& dataset);
		double dist_centroids(std::array<cluster<DIM>, K>& centroids, std::array<cluster<DIM>, K>& pre_centroids);
		std::array<cluster<DIM>, K> _centroids;
		std::array<cluster<DIM>, K> _pre_centroids;


	};

	template<size_t DIM, size_t K>
	inline void kmeans<DIM, K>::kmeans_init()
	{
		srand((size_t)time(NULL));
	}

	template<size_t DIM, size_t K>
	inline std::array<cluster<DIM>, K>& kmeans<DIM, K>::get_kmeans(std::vector<std::array<double, DIM>>& dataset, double threshold)
	{
		assert(dataset.size() != 0);
		init_centroids(dataset);
		while (true) {
			_pre_centroids = _centroids; //保存上一次的中心点
			_centroids.fill(cluster<DIM>());
			for (size_t data_index = 0; data_index < dataset.size(); data_index++) {
				size_t centroid_index = get_centroid(dataset[data_index], _pre_centroids);
				_centroids[centroid_index].count++;
				for (int i = 0; i < DIM; i++) {
					_centroids[centroid_index].node[i] += dataset[data_index][i];
				}
			}
			for (size_t centroid_index = 0; centroid_index < _centroids.size(); centroid_index++) {
				for (int i = 0; i < DIM; i++) {
					if(_centroids[centroid_index].count!=0){
					_centroids[centroid_index].node[i] = _centroids[centroid_index].node[i] / _centroids[centroid_index].count;
					}
				}
			}
			double dist = dist_centroids(_pre_centroids, _centroids);
			if (dist < threshold) {
				break;
			}
		}
		return _centroids;
	}

	template<size_t DIM, size_t K>
	inline size_t kmeans<DIM, K>::get_centroid(std::array<double, DIM>& node, std::array<cluster<DIM>, K>& centroids)
	{
		size_t centroid_index = 0;
		double min_distance = get_distance(node, centroids[0].node);
		for (size_t i = 0; i < K; i++) {
			if (0 == i) {
				continue;
			}
			double distance = get_distance(node, centroids[i].node);
			if (distance < min_distance) {
				min_distance = distance;
				centroid_index = i;
			}
		}
		return centroid_index;
	}

	template<size_t DIM, size_t K>
	inline double kmeans<DIM, K>::get_distance(std::array<double, DIM>& node_left, std::array<double, DIM>& node_right)
	{
		double result = 0;
		for (size_t i = 0; i < DIM; i++) {
			result += std::pow(node_left[i] - node_right[i], 2);
		}
		return std::sqrt(result);
	}

	template<size_t DIM, size_t K>
	inline void kmeans<DIM, K>::init_centroids(std::vector<std::array<double, DIM>>& dataset)
	{
		std::array<double, DIM> max_centroid = dataset[0];
		std::array<double, DIM> min_centroid = dataset[0];
		for (auto it = dataset.begin(); it != dataset.end(); it++) {
			for (int i = 0; i < DIM; i++) {
				max_centroid[i] = std::max(max_centroid[i], it->at(i));
				min_centroid[i] = std::min(min_centroid[i], it->at(i));
			}
		}
		std::array<double, DIM> base_centroid;
		for (size_t i = 0; i < DIM; i++) {
			base_centroid[i] = max_centroid[i] - min_centroid[i];
		}
		for (size_t means = 0; means < K; means++) {
			for (int i = 0; i < DIM; i++) {
				double d = ((double)rand()) / RAND_MAX;
				_centroids[means].node[i] = min_centroid[i] + base_centroid[i] * d;
			}
		}
		return;
	}

	template<size_t DIM, size_t K>
	inline double kmeans<DIM, K>::dist_centroids(std::array<cluster<DIM>, K>& centroids, std::array<cluster<DIM>, K>& pre_centroids)
	{
		double sum = 0;
		for (size_t means = 0; means < K; means++) {
			sum += get_distance(centroids[means].node, pre_centroids[means].node);
		}
		return std::sqrt(sum);;
	}


}


#endif // !K_MEANS_H