#include "MLP.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>




/*float calculate_accuracy(MLP *mlp, float labels[MAX_SAMPLES], int sample_count, float tolerance) {
    int correct_predictions = 0;
    for (int i = 0; i < sample_count; i++) {
        float prediction = mlp->layers[mlp->num_layers - 1].output[0];
        //printf("%f ",prediction);
        if (fabs(prediction - labels[i]) <= tolerance) { // fabs : valore assoluto
            correct_predictions++;
        }
    }
    return (float)correct_predictions / sample_count * 100.0;
}*/


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
   
    // Example input data
    float input_data[MAX_SAMPLES][MAX_FEATURES];
    float true_value[MAX_SAMPLES];

    // Read data from file
    const char *path = "./datasets/iris_dataset/iris_dataset.txt";
    int sample_count = read_data_from_file(path, MAX_FEATURES, 1, input_data, true_value);
    //sample_count = 3;
    
    // call the forward function for all the samples, and calculate the accuracy
    int correct_predictions = 0;
    for (int i = 0; i < sample_count; i++) {
        int prediction = forward(input_data[i][0], input_data[i][1], input_data[i][2], input_data[i][3]);
        if (prediction == true_value[i]) {
            correct_predictions++;
        }else{
            printf("Prediction: %d, True value: %f for input: %f %f %f %f\n", prediction, true_value[i], input_data[i][0], input_data[i][1], input_data[i][2], input_data[i][3]);
        }
    }
    float accuracy = (float)correct_predictions / sample_count * 100.0;
    printf("Accuracy: %.2f%%\n", accuracy);
}
