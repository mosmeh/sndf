#include <iostream>
#include <sstream>
#include <fstream>
#include <valarray>
#include <iterator>

#include <gflags/gflags.h>

#include "stochastic_decision_tree.h"

DEFINE_int32(seed, 1, "");
DEFINE_int32(depth, 10, "depth of tree");
DEFINE_string(dataset, "letter-recognition.data", "");
DEFINE_int32(train_examples, 16000, "");
DEFINE_int32(snapshot, 100, "takes snapshots every |snapshot| iterations");
DEFINE_int32(test_interval, 1, "tests every |test_interval| iterations");

std::tuple<float, float> evaluate(const StochasticDecisionTree& tree, const std::vector<std::pair<std::valarray<float>, int> >& examples) {
    float loss = 0, error = 0;
    for (auto& ex : examples) {
        const auto pred = tree.predictDeterminately(ex.first);

        std::valarray<float> target(0.0, NUM_CLASSES);
        target[ex.second] = 1;
        const auto r = pred - target;
        loss += (r * r).sum();

        const int m = std::distance(std::begin(pred), std::max_element(std::begin(pred), std::end(pred)));
        error += static_cast<float>(ex.second != m);
    }

    return std::make_tuple(loss / examples.size(), error / examples.size());
}

std::vector<std::pair<std::valarray<float>, int> > loadDataset(const std::string& filename) {
    std::vector<std::pair<std::valarray<float>, int> > examples;

    std::ifstream ifs(filename.c_str(), std::ios::in);
    std::string str, token;
    while (std::getline(ifs, str)) {
        std::istringstream iss(str);
        int label;
        std::valarray<float> features(0.0, NUM_FEATURES);
        for (int t = 0; std::getline(iss, token, ','); ++t) {
            if (t == 0) {
                label = static_cast<int>(token[0]) - 65;
            } else {
                features[t - 1] = std::stoi(token);
            }
        }
        examples.emplace_back(std::make_pair(features, label));
    }

    return std::move(examples);
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    std::mt19937 randomEngine(FLAGS_seed);

    std::cout << "loading dataset" << std::endl;
    auto dataset = loadDataset(std::string(FLAGS_dataset));
    std::cout << "#examples = " << dataset.size() << std::endl;
    std::shuffle(dataset.begin(), dataset.end(), randomEngine);

    std::vector<std::pair<std::valarray<float>, int> > trainExamples, testExamples;
    trainExamples.reserve(FLAGS_train_examples);
    testExamples.reserve(dataset.size() - FLAGS_train_examples);
    std::copy(dataset.begin(), dataset.begin() + FLAGS_train_examples, std::back_inserter(trainExamples));
    std::copy(dataset.begin() + FLAGS_train_examples, dataset.end(), std::back_inserter(testExamples));

    std::cout << "constructing decision tree" << std::endl;
    StochasticDecisionTree tree(FLAGS_depth, randomEngine);

    for (int i = 1;; ++i) {
        std::cout << "starting epoch #" << i << std::endl;

        std::cout << "training" << std::endl;
        std::shuffle(trainExamples.begin(), trainExamples.end(), randomEngine);

        tree.update(trainExamples);

        if (i % FLAGS_test_interval == 0) {
            std::cout << "testing" << std::endl;
            float loss, error;
            std::tie(loss, error) = evaluate(tree, trainExamples);
            std::cout << "train set: loss = " << loss << ", error = " << error << std::endl;
            std::tie(loss, error) = evaluate(tree, testExamples);
            std::cout << "test set:  loss = " << loss << ", error = " << error << std::endl;
        }

        if (i % FLAGS_snapshot == 0) {
            std::stringstream ss;
            ss << "iter" << i << ".model";
            std::cout << "saving" << std::endl;
            tree.saveModel(ss.str());
        }
    }

    return 0;
}
