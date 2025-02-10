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
    } Layer;

    std::vector<Layer> layers;

    std::vector<int> dims;

public:
    NN(const std::vector<int>& d, const std::vector<std::function<double(double)>> activation) {
        // initiatize layers
        dims = d;

        std::srand((unsigned int) time(0));
        
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

    void mutate(double rate) {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].weight += Eigen::MatrixXd::Random(dims[i + 1], dims[i]) * rate;
            layers[i].bias += Eigen::VectorXd::Random(dims[i + 1]) * rate;
        }
    }

    void visualize() {
        for (int i = 0; i < layers.size(); i++) {
            std::cout << layers[i].weight.reshaped(1, layers[i].weight.rows() * layers[i].weight.cols()) << '\n' << layers[i].bias << '\n' << std::endl;
        }
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

    //model.visualize();
    std::cout << model.predict(Eigen::Vector3d(1, -1, 0)) << std::endl;

    model.mutate(0.1);

    //model.visualize();

    std::cout << model.predict(Eigen::Vector3d(1, -1, 0)) << std::endl;
}