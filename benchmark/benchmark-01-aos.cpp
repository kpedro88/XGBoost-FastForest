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

#include <string>
#include <iterator>
template <class T, class O>
void printvec(const std::vector<T>& vec, O& out, const std::string& delim){
	if(!vec.empty()){
		//avoid trailing delim
		std::copy(vec.begin(),vec.end()-1,std::ostream_iterator<T>(out,delim.c_str()));
		//last element
		out << vec.back();
	}
}

TreeEnsembleResponseType evaluateBinary(const FastForest& forest, const FeatureType* array, bool debug) {
    TreeEnsembleResponseType out{defaultBaseResponse + forest.baseResponses_[0]};

	if(debug) std::cout << out << std::endl;
    for (int index : forest.rootIndices_) {
        do {
            auto r = forest.rightIndices_[index];
            auto l = forest.leftIndices_[index];
			if(debug) std::cout << index << " " << forest.cutIndices_[index] << " " << forest.cutValues_[index] << " " << forest.leftIndices_[index] << " " << forest.rightIndices_[index] << std::endl;
			if(debug) std::cout << array[forest.cutIndices_[index]];
            index = array[forest.cutIndices_[index]] > forest.cutValues_[index] ? r : l;
			if(debug) std::cout << " " << index << std::endl;
        } while (index > 0);
        out += forest.responses_[-index];
		if(debug) std::cout << index << " " << forest.responses_[-index] << std::endl;
    }
	if(debug) std::cout << out << std::endl;

    return out;
}

#define MAX_EVENTS 10
#define MAX_TREES 1
class DForest {
public:
	//based on FastForest::evaluate() and BDTree::parseTree()
	DForest(const FastForest& old) {
		baseResponses_ = old.baseResponses_;
		//loop through root nodes
#ifdef MAX_TREES
		for (int iRootIndex = 0; iRootIndex < MAX_TREES; ++iRootIndex) {
#else
		for (int iRootIndex = 0; iRootIndex < old.rootIndices_.size(); ++iRootIndex) {
#endif
			int index = old.rootIndices_[iRootIndex];
#ifdef MAX_TREES
			std::cout << "Head " << iRootIndex << ": " << index << ", " << old.cutIndices_[index] << ", " << old.cutValues_[index] << ", " << old.leftIndices_[index] << ", " << old.rightIndices_[index] << std::endl;
#endif
			convertTree(old, index, true);
#ifdef MAX_TREES
			printvec(nodes_,std::cout,","); std::cout << std::endl;
#endif
		}
	}
	void convertTree(const FastForest& old, int index, bool root=false){
		bool notLeaf = index > 0 or root;

		if(notLeaf) {
#ifdef MAX_TREES
			printvec(nodes_,std::cout,","); std::cout << std::endl;
#endif
			int thisidx = nodes_.size();
			if(root) rootIndices_.push_back(thisidx);
			nodes_.resize(nodes_.size()+4);
			nodes_[thisidx] = old.cutIndices_[index];
			nodes_[thisidx+1] = (int&)(old.cutValues_[index]);
#ifdef MAX_TREES
			std::cout << (float&)nodes_[thisidx+1] << std::endl;
#endif
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

	float evaluate(const float* features, bool debug) const {
		float sum{defaultBaseResponse + baseResponses_[0]};
		if(debug) std::cout << sum << std::endl;
		const int* node = nullptr;
		for(int index : rootIndices_){
			do {
				if(debug) std::cout << index << " " << nodes_[index] << " " << (float&)nodes_[index+1] << " " << nodes_[index+2] << " " << nodes_[index+3] << std::endl;
				node = &nodes_[index];
				if(debug) std::cout << features[nodes_[index]];
				index = features[*node] <= (float&)(*++node) ? *++node : *(node+2);
				if(debug) std::cout << " " << index << std::endl;
			} while (index>0);
			sum += responses_[-index];
			if(debug) std::cout << index << " " << responses_[-index] << std::endl;
		}
		if(debug) std::cout << sum << std::endl;
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
#ifdef MAX_TREES
	fastForest.rootIndices_.resize(MAX_TREES);
#endif
	DForest newForest(fastForest);

    //const int n = 100000;
    const int n = MAX_EVENTS;

    std::vector<float> input(5 * n);
    std::vector<float> scores(n);

	std::uint32_t seed = 10;
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> rand(-5,5);
    std::generate(input.begin(), input.end(), [&](){ return rand(rng); });

    clock_t begin = clock();
    for (int i = 0; i < n; ++i) {
//        scores[i] = 1. / (1. + std::exp(-fastForest(input.data() + i * 5)));
        scores[i] = 1. / (1. + std::exp(-evaluateBinary(fastForest,input.data() + i * 5, true)));
    }
    double average = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    std::cout << average << std::endl;

    clock_t end = clock();
    double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference 1: " << elapsedSecs << " s" << std::endl;

    std::vector<float> scores2(n);
    begin = clock();
    for (int i = 0; i < n; ++i) {
        scores2[i] = 1. / (1. + std::exp(-newForest.evaluate(input.data() + i * 5, true)));
    }
    average = std::accumulate(scores2.begin(), scores2.end(), 0.0) / scores2.size();
    std::cout << average << std::endl;

    end = clock();
    elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference 2: " << elapsedSecs << " s" << std::endl;
}
