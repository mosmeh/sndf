#include "stochastic_decision_tree.h"

StochasticDecisionTree::StochasticDecisionTree(const int depth, std::mt19937& randomEngine) :
    depth_(depth),
    randomEngine_(randomEngine),
    weights_(NUM_FEATURES * SPLIT_NODES),
    pi_(1.0 / NUM_CLASSES, LEAF_NODES * NUM_CLASSES),
    prevdWeights_(0.0, NUM_FEATURES * SPLIT_NODES),
    eta_(0.01, NUM_FEATURES * SPLIT_NODES),
    delta_(0.0, NUM_FEATURES * SPLIT_NODES) {

    std::normal_distribution<float> normal(0, 0.01);
    std::generate(std::begin(weights_), std::end(weights_), [&]() { return normal(randomEngine); });
}

void StochasticDecisionTree::loadModel(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    for (auto& x : weights_) {
        ifs.read(reinterpret_cast<char*>(&x), sizeof(float));
    }
    for (auto& x : pi_) {
        ifs.read(reinterpret_cast<char*>(&x), sizeof(float));
    }
    ifs.close();
}

void StochasticDecisionTree::saveModel(const std::string& filename) {
    std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
    for (auto& x : weights_) {
        ofs.write(reinterpret_cast<char*>(&x), sizeof(float));
    }
    for (auto& x : pi_) {
        ofs.write(reinterpret_cast<char*>(&x), sizeof(float));
    }
    ofs.close();
}

float StochasticDecisionTree::sigmoid(const float x) const {
    return 1 / (1 + std::exp(-x));
}

float StochasticDecisionTree::decisionFunction(const std::valarray<float>& features, const int n) const {
    const auto innerProduct = (features * weights_[std::slice(n * NUM_FEATURES, NUM_FEATURES, 1)]).sum();
    return sigmoid(innerProduct);
}

std::valarray<float> StochasticDecisionTree::predictDeterminately(const std::valarray<float>& features) const {
    std::valarray<float> d(SPLIT_NODES);
    for (int i = 0; i < SPLIT_NODES; ++i) {
        d[i] = decisionFunction(features, i);
    }

    std::valarray<float> mu(NODES);
    mu[0] = 1;
    for (int i = 0; i < SPLIT_NODES; ++i) {
        mu[2 * i + 1] = d[i] * mu[i];
        mu[2 * i + 2] = (1 - d[i]) * mu[i];
    }

    std::valarray<float> dist(0.0, NUM_CLASSES);
    for (int i = 0; i < LEAF_NODES; ++i) {
        dist += pi_[std::slice(i * NUM_CLASSES, NUM_CLASSES, 1)] * mu[i + SPLIT_NODES];
    }

    return dist;
}

std::tuple<int, float, std::valarray<float> > StochasticDecisionTree::predict(const std::valarray<float>& features) const {
    int n = 0;
    float mu = 1;
    while (n < SPLIT_NODES) {
        const float d = decisionFunction(features, n);
        const bool branchToLeft = std::bernoulli_distribution(d)(randomEngine_);
        mu *= branchToLeft ? d : (1 - d);
        n += n + 1 + (1 - static_cast<int>(branchToLeft));
    }

    return std::make_tuple(n - SPLIT_NODES, mu, pi_[std::slice((n - SPLIT_NODES) * NUM_CLASSES, NUM_CLASSES, 1)]);
}

int StochasticDecisionTree::leftNode(const int index) const {
    return 2 * index + 1;
}

int StochasticDecisionTree::rightNode(const int index) const {
    return 2 * index + 2;
}

void StochasticDecisionTree::update(const std::vector<std::pair<std::valarray<float>, int> >& examples) {
    std::vector<std::valarray<float> > mus(examples.size(), std::valarray<float>(LEAF_NODES)),
                                       ds(examples.size(), std::valarray<float>(SPLIT_NODES));
    for (unsigned int t = 0; t < examples.size(); ++t) {
        for (int i = 0; i < SPLIT_NODES; ++i) {
            ds[t][i] = decisionFunction(examples[t].first, i);
        }

        std::valarray<float> mu(NODES);
        mu[0] = 1;
        for (int i = 0; i < SPLIT_NODES; ++i) {
            mu[leftNode(i)] = ds[t][i] * mu[i];
            mu[rightNode(i)] = (1 - ds[t][i]) * mu[i];
        }

        mus[t] = mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)];
    }

    std::valarray<float> estimatedPi(1.0 / NUM_CLASSES, NUM_CLASSES * LEAF_NODES);
    for (int l = 0; l < 20; ++l) {
        std::valarray<float> tempPi(0.0, NUM_CLASSES * LEAF_NODES);
        for (unsigned int i = 0; i < examples.size(); ++i) {
            const int y = examples[i].second;
            const std::valarray<float> yProbs = mus[i] * static_cast<std::valarray<float> >(estimatedPi[std::slice(y, LEAF_NODES, NUM_CLASSES)]);
            tempPi[std::slice(y, LEAF_NODES, NUM_CLASSES)] += yProbs / (yProbs.sum() + EPS);
        }

        float diff = 0;
        for (int i = 0; i < LEAF_NODES; ++i) {
            std::valarray<float> dist = tempPi[std::slice(i * NUM_CLASSES, NUM_CLASSES, 1)];
            dist /= (dist.sum() + EPS);
            diff += std::abs(static_cast<std::valarray<float> >(estimatedPi[std::slice(i * NUM_CLASSES, NUM_CLASSES, 1)]) - dist).sum();
            estimatedPi[std::slice(i * NUM_CLASSES, NUM_CLASSES, 1)] = dist;
        }

        if (diff < 1e-5) {
            break;
        }
    }
    pi_ = estimatedPi;

    std::valarray<float> dWeights(0.0, SPLIT_NODES * NUM_FEATURES);
    for (unsigned int i_e = 0; i_e < examples.size(); ++i_e) {
        const int y = examples[i_e].second;

        const std::valarray<float> yProbs = mus[i_e] * static_cast<std::valarray<float> >(estimatedPi[std::slice(y, LEAF_NODES, NUM_CLASSES)]);

        std::valarray<float> a(NODES);
        a[std::slice(SPLIT_NODES, LEAF_NODES, 1)] = yProbs / (yProbs.sum() + EPS);

        for (int i_n = SPLIT_NODES - 1; i_n >= 0; --i_n) {
            a[i_n] = a[leftNode(i_n)] + a[rightNode(i_n)];
            const auto dLpdfi = ds[i_e][i_n] * a[rightNode(i_n)] - (1 - ds[i_e][i_n]) * a[leftNode(i_n)];
            dWeights[std::slice(i_n * NUM_FEATURES, NUM_FEATURES, 1)] += dLpdfi * examples[i_e].first;
        }
    }

    static const float etaPlus = 1.2, etaMinus = 0.5, deltaMax = 50, deltaMin = 1e-6;
    for (int i = 0; i < NUM_FEATURES * SPLIT_NODES; ++i) {
        if (prevdWeights_[i] * dWeights[i] > 0) {
            prevdWeights_[i] = dWeights[i];
            eta_[i] = std::min(eta_[i] * etaPlus, deltaMax);
            delta_[i] = (dWeights[i] > 0 ? 1 : -1) * eta_[i];
        } else if (prevdWeights_[i] * dWeights[i] < 0) {
            prevdWeights_[i] = 0;
            eta_[i] = std::max(eta_[i] * etaMinus, deltaMin);
            delta_[i] *= -1;
        } else {
            prevdWeights_[i] = dWeights[i];
            float sign = 0;
            if (dWeights[i] > 0) {
                sign = 1;
            } else if (dWeights[i] < 0) {
                sign = -1;
            }
            delta_[i] = sign * eta_[i];
        }
    }

    weights_ -= delta_;
}

