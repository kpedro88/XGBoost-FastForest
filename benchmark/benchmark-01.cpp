// compile with g++ -o benchmark-01 benchmark-01.cpp -lfastforest
//
// optimization flag does not matter because fastforest is already compiled

#include "fastforest.h"

#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <ctime>
#include "valgrind/install/include/valgrind/callgrind.h"

int main() {
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};

    const auto fastForest = fastforest::load_txt("model.txt", features);

    const int n = 100000;

    std::vector<float> input(5 * n);
    std::vector<double> scores(n);

	std::uint32_t seed = 10;
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> rand(-5,5);
    std::generate(input.begin(), input.end(), [&](){ return rand(rng); });

	CALLGRIND_START_INSTRUMENTATION;
	CALLGRIND_TOGGLE_COLLECT;
    clock_t begin = clock();
    for (int i = 0; i < n; ++i) {
        scores[i] = 1. / (1. + std::exp(-fastForest(input.data() + i * 5)));
    }
    double average = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    std::cout << average << std::endl;

    clock_t end = clock();
    double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Wall time for inference: " << elapsedSecs << " s" << std::endl;
	CALLGRIND_TOGGLE_COLLECT;
	CALLGRIND_STOP_INSTRUMENTATION;
	CALLGRIND_DUMP_STATS;
}
