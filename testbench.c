#include "MLP.h"
#include "utils.h"

int main() {
    // MLP parameters
    int number_of_features = 4;
    int input_neurons = number_of_features;
    int layer_sizes[] = {10, 100, 10, 1};
    int activations[] = {0, 0, 0, 4}; // 0 per sigmoid, 4 per linear
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    int epochs = 2;
    float learning_rate = 0.001;

    // Example input data
    float input_data[MAX_SAMPLES][MAX_FEATURES] = {{0.5, 0.2, 0.1, 0.7}};
    float true_value[MAX_SAMPLES] = {1.0};

    MLP *mlp = create_mlp(input_neurons, num_layers, layer_sizes, activations, 0); // 0 per MEAN_SQUARED_ERROR
    train(mlp, input_data, true_value, epochs, learning_rate);

    return 0;
}