// compile with g++ -O2 -o benchmark-01-aos benchmark-01-aos.cpp -lfastforest

#include "../include/fastforest.h"

#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <ctime>
#include <vector>
#include "valgrind/callgrind.h"

//from:
//https://github.com/kpedro88/Analysis/blob/SVJ2018/KMVA/BDTree.h
//https://github.com/kpedro88/cmssw/commit/909ae926b7b13ff3887250fc0fd7a02446dc121c

using namespace fastforest;

#define  BRANCHLESS_IF_ELSE(f,x,y)  (((x) & -((typeof(x))!!(f))) | \
                                    ((y) & -((typeof(y)) !(f))))

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
/*
				//version 1
				index = features[*node] <= (float&)(*++node) ? *++node : *(node+2);
*/
/*
				//version 2
				auto l = *(node+2);
				auto r = *(node+3);
				index = features[ind] <= (float&)cut ? l : r;
*/
/*
				//version 3
				auto fea = features[*node];
				auto cut = *++node;
				auto l = *++node;
				auto r = *++node;
				index = fea > (float&)cut ? r : l;
*/
/*
				//version 4
				auto m = features[*node] > (float&)(*++node);
				index = (int(!m)*(*++node)) + (int(m)*(*++node));
*/
/*
				//version 5
				index = BRANCHLESS_IF_ELSE(features[*node] <= (float&)(*(node+1)), *(node+2), *(node+3));
*/
				index = *(node+2+(features[*node] > (float&)(*(node+1))));
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

#ifdef TESTFF
    clock_t begin = clock();
    for (int i = 0; i < n; ++i) {
        scores[i] = 1. / (1. + std::exp(-fastForest(input.data() + i * 5)));
    }
    double average = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    std::cout << average << std::endl;

    clock_t end = clock();
    double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference 1: " << elapsedSecs << " s" << std::endl;
#endif

    std::vector<float> scores2(n);
	CALLGRIND_START_INSTRUMENTATION;
	CALLGRIND_TOGGLE_COLLECT;
    clock_t begin2 = clock();
    for (int i = 0; i < n; ++i) {
        scores2[i] = 1. / (1. + std::exp(-newForest.evaluate(input.data() + i * 5)));
    }
    float average2 = std::accumulate(scores2.begin(), scores2.end(), 0.0) / scores2.size();
    std::cout << average2 << std::endl;

    clock_t end2 = clock();
    double elapsedSecs2 = double(end2 - begin2) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference 2: " << elapsedSecs2 << " s" << std::endl;
	CALLGRIND_TOGGLE_COLLECT;
	CALLGRIND_STOP_INSTRUMENTATION;
	CALLGRIND_DUMP_STATS;
}
