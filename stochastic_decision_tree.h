#ifndef STOCHASTIC_DECISION_TREE_H_
#define STOCHASTIC_DECISION_TREE_H_

#include <vector>
#include <valarray>
#include <random>
#include <fstream>

const int NUM_CLASSES = 26;
const int NUM_FEATURES = 16;

class StochasticDecisionTree {
public:
    StochasticDecisionTree(const int depth, std::mt19937& randomEngine);

    std::tuple<int, float, std::valarray<float> > predict(const std::valarray<float>& features) const;
    std::valarray<float> predictDeterminately(const std::valarray<float>& features) const;
    void update(const std::vector<std::pair<std::valarray<float>, int> >& examples);
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

protected:
    const float EPS = 1e-7;
    const int depth_;

    const int LEAF_NODES = std::pow(2, depth_);
    const int SPLIT_NODES = std::pow(2, depth_) - 1;
    const int NODES = LEAF_NODES + SPLIT_NODES;

    std::mt19937& randomEngine_;
    std::valarray<float> weights_, pi_, prevdWeights_, eta_, delta_;

    float sigmoid(const float x) const;
    float decisionFunction(const std::valarray<float>& features, const int n) const;
    int leftNode(const int index) const;
    int rightNode(const int index) const;
};

#endif

