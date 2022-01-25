// compile with g++ -O2 -o benchmark-01-aos benchmark-01-aos.cpp -lfastforest

#include "../include/fastforest.h"

#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <ctime>
#include <vector>

//from:
//https://github.com/kpedro88/Analysis/blob/SVJ2018/KMVA/BDTree.h
//https://github.com/kpedro88/cmssw/commit/909ae926b7b13ff3887250fc0fd7a02446dc121c

using namespace fastforest;

class DForest {
public:
	//based on FastForest::evaluate() and BDTree::parseTree()
	DForest(const FastForest& old) {
		baseResponses_ = old.baseResponses_;
		//loop through root nodes
		for (int iRootIndex = 0; iRootIndex < old.rootIndices_.size(); ++iRootIndex) {
			int index = old.rootIndices_[iRootIndex];
			convertTree(old, index, true);
		}
	}
	void convertTree(const FastForest& old, int index, bool root=false){
		bool notLeaf = index > 0 or root;

		if(notLeaf) {
			int thisidx = nodes_.size();
			if(root) rootIndices_.push_back(thisidx);
			nodes_.resize(nodes_.size()+4);
			nodes_[thisidx] = old.cutIndices_[index];
			nodes_[thisidx+1] = (int&)(old.cutValues_[index]);
			//convert children recursively
			int left = old.leftIndices_[index];
			nodes_[thisidx+2] = left <= 0 ? -responses_.size() : nodes_.size();
			convertTree(old, left);
			int right = old.rightIndices_[index];
			nodes_[thisidx+3] = right <= 0 ? -responses_.size() : nodes_.size();
			convertTree(old, right);
		}
		else {
			responses_.push_back(old.responses_[-index]);
		}
	}

	float evaluate(const float* features) const {
		float sum{defaultBaseResponse + baseResponses_[0]};
		const int* node = nullptr;
		for(int index : rootIndices_){
			do {
				node = &nodes_[index];
				index = features[*node] <= (float&)(*++node) ? *++node : *(node+2);
			} while (index>0);
			sum += responses_[-index];
		}
		return sum;
	}

	std::vector<int> rootIndices_;
	//"node" layout: index, cut, left, right
	std::vector<int> nodes_;
	std::vector<float> responses_;
	std::vector<float> baseResponses_;
};

int main() {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    auto fastForest = fastforest::load_txt("model.txt", features);
	DForest newForest(fastForest);

    const int n = 100000;

    std::vector<float> input(5 * n);
    std::vector<float> scores(n);

	std::uint32_t seed = 10;
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> rand(-5,5);
    std::generate(input.begin(), input.end(), [&](){ return rand(rng); });

    clock_t begin = clock();
    for (int i = 0; i < n; ++i) {
        scores[i] = 1. / (1. + std::exp(-fastForest(input.data() + i * 5)));
    }
    double average = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    std::cout << average << std::endl;

    clock_t end = clock();
    double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference 1: " << elapsedSecs << " s" << std::endl;

    std::vector<float> scores2(n);
    begin = clock();
    for (int i = 0; i < n; ++i) {
        scores2[i] = 1. / (1. + std::exp(-newForest.evaluate(input.data() + i * 5)));
    }
    average = std::accumulate(scores2.begin(), scores2.end(), 0.0) / scores2.size();
    std::cout << average << std::endl;

    end = clock();
    elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference 2: " << elapsedSecs << " s" << std::endl;
}
