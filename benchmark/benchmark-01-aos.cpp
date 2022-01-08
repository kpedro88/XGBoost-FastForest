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
	DNode(unsigned i, double c, int l=0, int r=0) : index_(i), cut_(c), left_(l), right_(r) {}
	unsigned index_;
	float cut_;
	int left_;
	int right_;
};

class DTree {
	public:
		DTree() {}
		virtual ~DTree() {}
		inline float decision(const float* features) const {
			if(nodes_.empty()) return vres_[0]; //single leaf tree case
			int index = 0;
			do {
				auto l = nodes_[index].left_;
				auto r = nodes_[index].right_;
				index = features[nodes_[index].index_] <= nodes_[index].cut_ ? l : r;
			} while (index>0);
			return vres_[-index];
		}
		std::vector<DNode> nodes_;
		std::vector<float> vres_;
};

using namespace fastforest;

class DForest {
public:
	//based on FastForest::evaluate() and BDTree::parseTree()
	DForest(const FastForest& old) {
		//loop through root nodes
		trees_.resize(old.rootIndices_.size());
		for (int iRootIndex = 0; iRootIndex < old.rootIndices_.size(); ++iRootIndex) {
			int index = old.rootIndices_[iRootIndex];
			bool isSingleLeafTree = index < 0;
			convertTree(old, index, isSingleLeafTree, trees_[iRootIndex]);
		}
	}
	void convertTree(const FastForest& old, int index, bool isSingleLeafTree, DTree& tree){
		bool notLeaf = index > 0;
		if(isSingleLeafTree) {
			index++;
			notLeaf = false;
		}

		if(notLeaf) {
			int thisidx = tree.nodes_.size();
			tree.nodes_.emplace_back(old.cutIndices_[index],old.cutValues_[index]);
			//convert children recursively
			int left = old.leftIndices_[index];
			tree.nodes_[thisidx].left_ = left < 0 ? -tree.vres_.size() : tree.nodes_.size();
			convertTree(old, left, false, tree);
			int right = old.rightIndices_[index];
			tree.nodes_[thisidx].right_ = right < 0 ? -tree.vres_.size() : tree.nodes_.size();
			convertTree(old, right, false, tree);
		}
		else {
			tree.vres_.push_back(old.responses_[-index]);
		}
	}

	float evaluate(const float* features){
		float sum = 0.;
		for(const auto& tree : trees_){
			sum += tree.decision(features);
		}
		return sum;
	}

	std::vector<DTree> trees_;
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
