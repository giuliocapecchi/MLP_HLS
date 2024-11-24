#include "MLP.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void print_mlp(MLP *mlp) {
    printf("*-----------------------------*\n");
    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = &mlp->layers[i];
        printf("Layer %d:\n", i);
        printf("Weights:\n");
        for (int j = 0; j < layer->input_size; j++) {
            for (int k = 0; k < layer->output_size; k++) {
                printf("%f ", layer->weights[j][k]);
            }
            printf("\n");
        }
        printf("Biases:\n");
        for (int j = 0; j < layer->output_size; j++) {
            printf("%f ", layer->biases[j]);
        }
        printf("\n");
    }
    printf("\n*-----------------------------*\n");
}


float calculate_accuracy(MLP *mlp, float labels[MAX_SAMPLES], int sample_count, float tolerance) {
    int correct_predictions = 0;
    for (int i = 0; i < sample_count; i++) {
        float prediction = mlp->layers[mlp->num_layers - 1].output[0];
        //printf("%f ",prediction);
        if (fabs(prediction - labels[i]) <= tolerance) { // fabs : valore assoluto
            correct_predictions++;
        }
    }
    return (float)correct_predictions / sample_count * 100.0;
}


int read_data_from_file(const char *path, int num_features, int label_size, float input_data[MAX_SAMPLES][MAX_FEATURES], float true_value[MAX_SAMPLES]) {
    FILE *file = fopen(path, "r");
    if (!file) {
        perror("Failed to open file");
        return -1;
    }

    int sample_count = 0;
    while (fscanf(file, "%f", &input_data[sample_count][0]) != EOF) {
        for (int i = 1; i < num_features; i++) {
            fscanf(file, "%f", &input_data[sample_count][i]);
        }
        for (int j = 0; j < label_size; j++) {
            fscanf(file, "%f", &true_value[sample_count]);
        }
        sample_count++;
        if (sample_count >= MAX_SAMPLES) {
            break;
        }
    }

    fclose(file);
    return sample_count;
}

int main() {
    // MLP parameters
    int epochs = 10;
    int number_of_features = 1;
    int input_neurons = number_of_features;
    int layer_sizes[] = {1};
    int activations[] = {4}; // 0 : sigmoid, 1: ReLu, 2: LeakyReLu, 3: HeavySide, 4 : linear
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    float learning_rate = 0.001;

    // Example input data
    float input_data[MAX_SAMPLES][MAX_FEATURES];
    float true_value[MAX_SAMPLES];

    // Read data from file
    const char *path = "./datasets/linear_regression/linear_regression.txt";
    
    int sample_count = read_data_from_file(path, number_of_features, 1, input_data, true_value);
    if (sample_count == -1) {
        printf("Failed to read data from file.\n");
        return 1;
    }

    // HLS pragmas
    //partizionano l'array lungo la prima dimensione (accesso parallelo agli elementi)
    #pragma HLS ARRAY_PARTITION variable=input_data complete dim=1 
    #pragma HLS ARRAY_PARTITION variable=true_value complete dim=1

    printf("Creating an MLP with %d input neurons and %d layers in %d epochs...\n", input_neurons, num_layers, epochs);
    MLP *mlp = create_mlp(input_neurons, num_layers, layer_sizes, activations, 0); // 0 : MEAN_SQUARED_ERROR, 1 : BINARY_CROSS_ENTROPY

    // Train for each epoch and print the results
    for (int epoch = 0; epoch < epochs; epoch++) {
        train(mlp, input_data, true_value, sample_count, learning_rate);
        float loss = meanSquaredError(mlp->layers[mlp->num_layers - 1].output, true_value, sample_count);
        float accuracy = calculate_accuracy(mlp, true_value, sample_count, 0.1);
        printf("Epoch %d\tLoss: %f\tAccuracy: %f\n", epoch + 1, loss, accuracy);
    }
    printf("Training completed.\n");
    print_mlp(mlp);
    return 0;
}