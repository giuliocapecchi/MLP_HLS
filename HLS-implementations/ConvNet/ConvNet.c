#include <stdlib.h>
#include <string.h>
#include "ConvNet.h"


float reLu(float x) {
    #pragma HLS INLINE
    return x > 0 ? x : 0;
}

// float tanh(float x) {
//     #pragma HLS INLINE
//     return (2 / (1 + exp(-2 * x))) - 1;
//}

ConvNet convnet = {
    .num_layers = 3,
    .layers = {
        // conv1
        {
            .weights = {
                // ...initialize weights...
            },
            .biases = {
                // ...initialize biases...
            }
        },
        // conv2
        {
            .weights = {
                // ...initialize weights...
            },
            .biases = {
                // ...initialize biases...
            }
        },
        // fc
        {
            .weights = {
                // ...initialize weights...
            },
            .biases = {
                // ...initialize biases...
            }
        }
    }
};

int forward(float input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS]) {
    const int input_sizes[4] = {INPUT_CHANNELS, CONV1_OUTPUT_CHANNELS, CONV2_OUTPUT_CHANNELS, FC_OUTPUT_SIZE};
    const int num_layers = 3;

    float current_input[MAX_NEURONS];
    float next_input[MAX_NEURONS];

    // Flatten input
    for (int i = 0; i < INPUT_HEIGHT; i++) {
        for (int j = 0; j < INPUT_WIDTH; j++) {
            for (int k = 0; k < INPUT_CHANNELS; k++) {
                current_input[i * INPUT_WIDTH * INPUT_CHANNELS + j * INPUT_CHANNELS + k] = input[i][j][k];
            }
        }
    }

    for (int i = 0; i < num_layers; i++) { // this loop performs the forward pass
        #pragma HLS UNROLL
        Layer *layer = &convnet.layers[i];
        for (int j = 0; j < input_sizes[i + 1]; j++) {
            float sum = layer->biases[j];
            
            for (int k = 0; k < input_sizes[i]; k++) {
                sum += layer->weights[j][k] * current_input[k];
            }
            next_input[j] = reLu(sum);
        }

        for (int j = 0; j < input_sizes[i + 1]; j++) {
            current_input[j] = next_input[j];
        }
    }

    int max_index = 0;
    float max = current_input[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        #pragma HLS UNROLL
        if (current_input[i] > max) {
            max = current_input[i];
            max_index = i;
        }
    }
    return max_index;
}
