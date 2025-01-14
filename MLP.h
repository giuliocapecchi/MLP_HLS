#ifndef MLP_H
#define MLP_H

#define MAX_LAYERS 3                // max number of layers
#define MAX_NEURONS 100             // max neurons per layer
#define MAX_SAMPLES 1000            // max number of samples
#define MAX_FEATURES 4              // max number of features per sample
#define NUM_CLASSES 3               // number of classes
#define LEARNING_RATE 0.01          // learning rate
#define OUTPUT_SIZE 3               // output size

#include "utils.h"

/*------------------------ Data Structures ------------------------*/

typedef struct {
    float weights[MAX_NEURONS][MAX_NEURONS];  // weights of the layer
    float biases[MAX_NEURONS];   // biases of the layer
    float output[MAX_NEURONS];   // output of the layer
} Layer;

typedef struct {
    int num_layers;              // number of layers
    Layer layers[MAX_LAYERS];    // layers of the MLP (array of layers)
} MLP;

/*-------------------------- Functions ---------------------------*/

int forward(float input0, float input1, float input2, float input3);

#endif // MLP_H