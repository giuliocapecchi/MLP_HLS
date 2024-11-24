#include "MLP.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void print_mlp(MLP *mlp) {
    printf("Printing the MLP...\n");
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
    printf("Final output:\n");
    for (int i = 0; i < mlp->layers[mlp->num_layers - 1].output_size; i++) {
        printf("%f ", mlp->layers[mlp->num_layers - 1].output[i]);
    }
    printf("\n");
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
    int number_of_features = 1;
    int input_neurons = number_of_features;
    int layer_sizes[] = {1};
    int activations[] = {4}; // 0 : sigmoid, 4 : linear
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    int epochs = 1000;
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


    printf("input_data:\n");
    for (int i = 0; i < sample_count; i++) {
        for (int j = 0; j < number_of_features; j++) {
            printf("%f ", input_data[i][j]);
        }
    }
    printf("\ntrue_value:\n");
    for (int i = 0; i < sample_count; i++) {
        printf("%f ", true_value[i]);
    }
    printf("\n");
    

    // HLS pragmas
    //partizionano l'array lungo la prima dimensione (accesso parallelo agli elementi)
    #pragma HLS ARRAY_PARTITION variable=input_data complete dim=1 
    #pragma HLS ARRAY_PARTITION variable=true_value complete dim=1

    MLP *mlp = create_mlp(input_neurons, num_layers, layer_sizes, activations, 0); // 0 : MEAN_SQUARED_ERROR, 1 : BINARY_CROSS_ENTROPY

    // Print initial weights and biases
    printf("Initial weights and biases:\n");
    print_mlp(mlp);

    // Train for each epoch and print the results
    for (int epoch = 0; epoch < epochs; epoch++) {
        train(mlp, input_data, true_value, sample_count, learning_rate);
        printf("After epoch %d:\n", epoch + 1);
        print_mlp(mlp);
    }

    printf("Training completed.\n");
    return 0;
}