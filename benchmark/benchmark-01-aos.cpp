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

struct DNode {
	DNode(unsigned i, float c, int l=0, int r=0) : index_(i), cut_(c), left_(l), right_(r) {}
	unsigned index_;
	float cut_;
	int left_;
	int right_;
};

using namespace fastforest;

class DForest {
public:
	//based on FastForest::evaluate() and BDTree::parseTree()
	DForest(const FastForest& old) {
		//loop through root nodes
		for (int iRootIndex = 0; iRootIndex < old.rootIndices_.size(); ++iRootIndex) {
			int index = old.rootIndices_[iRootIndex];
			convertTree(old, index, true);
		}
	}
	void convertTree(const FastForest& old, int index, bool root=false){
		bool notLeaf = index > 0;

		if(notLeaf) {
			int thisidx = nodes_.size();
			if(root) rootIndices_.push_back(thisidx);
			nodes_.emplace_back(old.cutIndices_[index],old.cutValues_[index]);
			//convert children recursively
			int left = old.leftIndices_[index];
			nodes_[thisidx].left_ = left < 0 ? -responses_.size() : nodes_.size();
			convertTree(old, left);
			int right = old.rightIndices_[index];
			nodes_[thisidx].right_ = right < 0 ? -responses_.size() : nodes_.size();
			convertTree(old, right);
		}
		else {
			responses_.push_back(old.responses_[-index]);
		}
	}

	float evaluate(const float* features){
		float sum = 0.;
		for(int i = 0; i < baseResponses_.size(); ++i){
			sum += baseResponses_[i];
		}
		for(int index : rootIndices_){
			do {
				const auto& node = nodes_[index];
				auto l = node.left_;
				auto r = node.right_;
				index = features[node.index_] <= node.cut_ ? l : r;
			} while (index>0);
			sum += responses_[-index];
		}
		return sum;
	}

	std::vector<int> rootIndices_;
	std::vector<DNode> nodes_;
	std::vector<float> responses_;
	std::vector<float> baseResponses_;
};

int main() {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    const auto fastForest = fastforest::load_txt("model.txt", features);
	DForest newForest(fastForest);

    const int n = 100000;

    std::vector<float> input(5 * n);
    std::vector<float> scores(n);

    std::generate(input.begin(), input.end(), std::rand);
    for (auto& x : input) {
        x = float(x) / RAND_MAX * 10 - 5;
    }

    clock_t begin = clock();
    for (int i = 0; i < n; ++i) {
        scores[i] = 1. / (1. + std::exp(-newForest.evaluate(input.data() + i * 5)));
    }
    double average = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    std::cout << average << std::endl;

    clock_t end = clock();
    double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference: " << elapsedSecs << " s" << std::endl;
}
