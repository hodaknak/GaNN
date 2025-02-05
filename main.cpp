#include <iostream>
#include <Eigen/Dense>

#include <vector>
#include <algorithm>
#include <functional>

class NN {
private:
    // each linear layer stores weights, bias, and activation function of connection
    typedef struct {
        Eigen::MatrixXd weight;
        Eigen::VectorXd bias;
        std::function<double(double)> activation;
    } layer;

    std::vector<layer> layers;

public:
    NN(const std::vector<int>& dims, const std::vector<std::function<double(double)>> activation) {
        // initiatize layers
        for (int i = 0; i < dims.size() - 1; i++) {
            layers.push_back({
                Eigen::MatrixXd::Random(dims[i + 1], dims[i]),
                Eigen::VectorXd::Random(dims[i + 1]),
                activation[i]
            });
        }
    }

    Eigen::VectorXd predict(const Eigen::VectorXd& input) {
        // feed forward through the network

        Eigen::VectorXd y = std::move(input);
        for (int i = 0; i < layers.size(); i++) {
            // left multiply the layer matrix, add bias to resulting vector, and apply activation
            y = (layers[i].weight * y + layers[i].bias).unaryExpr(layers[i].activation);
        }

        return y;
    }

    static double relu(double x) {
        return std::max(0.0, x);
    }

    static double sigmoid(double x) {
        return x / (2 + 2 * std::abs(x)) + 0.5;
    }
};

int main() {
    NN model= NN({3, 20, 4}, {NN::relu, NN::sigmoid});

    //std::cout << Eigen::MatrixXd(3, 3) * Eigen::VectorXd(1, 1, 1) << std::endl;

    std::cout << model.predict(Eigen::Vector3d(1, -1, 0)) << std::endl;
}