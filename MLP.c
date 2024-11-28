#include "MLP.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>


MLP mlp = {
        .num_layers = 3,
        .layers = {
            // fc1
            {
                .weights = {
                    {0.099288, -0.452716, 0.957605, 0.517334},
                    {0.220560, -0.666361, 0.772435, 0.389321},
                    {0.957434, 0.931137, -0.088062, -0.886874},
                    {0.171310, -0.261900, -0.481757, -0.173231},
                    {-0.329190, -0.124102, 0.428382, -0.030107},
                    {0.181957, -0.538837, 0.682200, 0.713888},
                    {0.144945, 0.507604, -0.269793, -0.199517},
                    {-0.075747, 0.202625, -0.398569, -0.076711},
                    {0.101007, -0.398583, -0.366854, -0.304911},
                    {-0.525624, -0.124397, 0.663196, 1.085638}
                },
                .biases = {-0.780636, -0.114756, 1.137292, 0.116033, -0.233900, -0.257343, 0.012394, -0.086546, -0.312638, -0.667115}
            },
            // fc2
            {
                .weights = {
                    {-0.435560, -0.538333, 0.253035, 0.120499, 0.098796, 0.016922, 0.362046, -0.070930, 0.207683, -0.152756},
                    {-0.295931, 0.123051, -0.294462, 0.027270, 0.159673, -0.088673, 0.124290, 0.177104, -0.037433, -0.142208},
                    {-0.299856, 0.042192, -0.052551, 0.167064, -0.207485, 0.039520, -0.288037, 0.087825, -0.047375, -0.220476},
                    {0.228607, 0.308247, -0.321292, -0.028800, -0.093888, 0.400650, 0.039870, -0.184433, 0.010677, 0.760814},
                    {0.051853, -0.047708, -0.241395, 0.217187, 0.014528, -0.315055, -0.225099, 0.089136, -0.281209, -0.110815},
                    {0.730003, 0.647504, -0.068370, 0.202805, -0.028918, 0.331941, -0.374136, -0.101397, -0.074116, 0.472928},
                    {-0.591532, -0.205189, 0.809988, -0.033138, 0.190923, -0.071424, 0.048885, 0.272477, -0.167278, -0.746578},
                    {0.086213, -0.317306, -0.151554, 0.234929, 0.126355, -0.012326, 0.094182, -0.186780, 0.060410, -0.014026},
                    {0.712435, 0.435129, 0.125985, 0.231775, 0.269586, 0.848233, -0.540500, -0.059259, 0.241100, 0.581109},
                    {-0.498703, -0.301602, 0.832068, -0.029463, -0.160015, -0.421864, 0.481599, 0.196709, 0.155504, -0.502033}
                },
                .biases = {0.391001, -0.102510, 0.112756, -0.503014, 0.308852, -0.300739, 0.549250, -0.227250, 0.216680, 0.251347}
            },
            // fc3
            {
                .weights = {
                    {0.723252, -0.216710, -0.209677, -0.236799, 0.125862, -0.505804, 0.557877, 0.228974, -0.673829, 0.511271},
                    {-0.780301, 0.307417, 0.099260, -0.792102, -0.230814, 0.036830, 0.645035, -0.187485, 0.132777, 0.106330},
                    {-0.166297, 0.061751, 0.221582, 0.440270, -0.070659, 0.727096, -0.954511, 0.135507, 0.253526, -0.723126}
                },
                .biases = {-0.052471, 0.534004, -0.171671}
            }
        }
    };



int forward(float input0, float input1, float input2, float input3) {
    // Inizializza array statico per le dimensioni degli input
    const int input_sizes[4] = {4, 10, 10, 3}; // Numero di feature e neuroni dei layer successivi
    const int num_layers = 3; // Numero di layer

    float current_input[MAX_NEURONS];
    float next_input[MAX_NEURONS];

    // Inizializza l'input con le feature
    current_input[0] = input0;
    current_input[1] = input1;
    current_input[2] = input2;
    current_input[3] = input3;

    //for (int i = 0; i < num_layers; i++) {
    Layer *layer = &mlp.layers[0]; 
    int i = 0;

    for (int j = 0; j < input_sizes[i + 1]; j++) {
        float sum = layer->biases[j];
        for (int k = 0; k < input_sizes[i]; k++) {
            sum += layer->weights[j][k] * current_input[k];
        }
        next_input[j] = reLu(sum); // Funzione di attivazione
    }

    // Copia l'output come input per il prossimo layer
    for (int j = 0; j < input_sizes[i + 1]; j++) {
        current_input[j] = next_input[j];
    }

    // second layer
    layer = &mlp.layers[1]; 
    i = 1;

    for (int j = 0; j < input_sizes[i + 1]; j++) {
        float sum = layer->biases[j];
        for (int k = 0; k < input_sizes[i]; k++) {
            sum += layer->weights[j][k] * current_input[k];
        }
        next_input[j] = reLu(sum); // Funzione di attivazione
    }

    // Copia l'output come input per il prossimo layer
    for (int j = 0; j < input_sizes[i + 1]; j++) {
        current_input[j] = next_input[j];
    }

    // third layer
    layer = &mlp.layers[2]; 
    i = 2;

    for (int j = 0; j < input_sizes[i + 1]; j++) {
        float sum = layer->biases[j];
        for (int k = 0; k < input_sizes[i]; k++) {
            sum += layer->weights[j][k] * current_input[k];
        }
        next_input[j] = reLu(sum); // Funzione di attivazione
    }

    // Copia l'output come input per il prossimo layer
    for (int j = 0; j < input_sizes[i + 1]; j++) {
        current_input[j] = next_input[j];
    }


    //}

    // Trova l'indice dell'output massimo (argmax)
    /*int max_index = 0;
    for (int i = 1; i < input_sizes[num_layers]; i++) {
        #pragma HLS PIPELINE II=1
        if (current_input[i] > current_input[max_index]) {
            max_index = i;
        }
    }*/

    int max_index = 0;
    i = 1;
    if (current_input[i] > current_input[max_index]) {
            max_index = i;
    }

    i = 2;
    if (current_input[i] > current_input[max_index]) {
            max_index = i;
    }

    i = 3;
    if (current_input[i] > current_input[max_index]) {
            max_index = i;
    }


    return max_index;
}



// // choose an activation function
// float activate(int function_id, float x) {
//     switch (function_id) {
//         case 0: return sigmoid(x);
//         case 1: return reLu(x);
//         case 2: return leakyReLu(x);
//         case 3: return heavySide(x);
//         case 4: return linear(x);
//         default: return x;
//     }
// }

// // choose the derivative of an activation function
// float activate_derivative(int function_id, float x) {
//     switch (function_id) {
//         case 0: return sigmoid_derivative(x);
//         case 1: return reLu_derivative(x);
//         case 2: return leakyReLu(x); // TODO: Leaky ReLu derivative
//         case 3: return heavySide(x); // TODO: Heavy side derivative
//         case 4: return linear_derivative();
//         default: return 1.0;
//     }
// }


// MLP* create_mlp(int input_neurons, int num_layers, int *layer_sizes, int *activations, int loss_function) {
//     static MLP mlp;  // static declare to avoid malloc 
//     mlp.num_layers = num_layers;
//     mlp.loss_function = loss_function;

//     for (int i = 0; i < num_layers; i++) {
//         mlp.layers[i].input_size = (i == 0) ? input_neurons : layer_sizes[i - 1];
//         mlp.layers[i].output_size = layer_sizes[i];
//         mlp.layers[i].activation_function = activations[i];

//         // Initialize weights and biases
//         for (int j = 0; j < mlp.layers[i].output_size; j++) {
//             mlp.layers[i].biases[j] = 0.0f;
//             for (int k = 0; k < mlp.layers[i].input_size; k++) {
//                 mlp.layers[i].weights[j][k] = ((float)rand() / RAND_MAX) - 0.5f; // Random initialization
//                 //mlp.layers[i].weights[j][k] = 0.0f; // Zero initialization
//             }
//         }
//     }

//     return &mlp;
// }



// calculate the error of the output layer
// void calculate_output_error(MLP *mlp, float *predicted, float *true_value, int loss_function) {
//     Layer *output_layer = &mlp->layers[mlp->num_layers - 1];
//     for (int i = 0; i < output_layer->output_size; i++) {
//         float error = predicted[i] - true_value[i];
//         if (loss_function == 0) { // MEAN_SQUARED_ERROR
//             output_layer->errors[i] = error * activate_derivative(output_layer->activation_function, predicted[i]);
//         } else if (loss_function == 1) { // BINARY_CROSS_ENTROPY
//             // TODO : implement binary cross entropy
//         }
//     }
// }

// backpropagation
// void backpropagate(MLP *mlp, float *input, float *true_value, float learning_rate) {
//     calculate_output_error(mlp, mlp->layers[mlp->num_layers - 1].output, true_value, mlp->loss_function);

//     for (int i = mlp->num_layers - 1; i >= 0; i--) {
//         Layer *layer = &mlp->layers[i];
//         float prev_output[MAX_NEURONS];

//         // copy the input in prev_output
//         if (i == 0) {
//             for (int j = 0; j < layer->input_size; j++) {
//                 prev_output[j] = input[j];
//             }
//         } else {
//             for (int j = 0; j < layer->input_size; j++) {
//                 prev_output[j] = mlp->layers[i - 1].output[j];
//             }
//         }

//         for (int j = 0; j < layer->output_size; j++) {
//             for (int k = 0; k < layer->input_size; k++) {
//                 layer->weights[j][k] -= learning_rate * layer->errors[j] * prev_output[k];
//             }
//             layer->biases[j] -= learning_rate * layer->errors[j];
//         }

//         if (i > 0) {
//             Layer *prev_layer = &mlp->layers[i - 1];
//             for (int j = 0; j < prev_layer->output_size; j++) {
//                 float error_sum = 0.0f;
//                 for (int k = 0; k < layer->output_size; k++) {
//                     error_sum += layer->weights[k][j] * layer->errors[k];
//                 }
//                 prev_layer->errors[j] = error_sum * activate_derivative(prev_layer->activation_function, prev_layer->output[j]);
//             }
//         }
//     }
// }

// training for a single epoch
// void train(MLP *mlp, float features[MAX_SAMPLES][MAX_FEATURES], float labels[MAX_SAMPLES], int sample_count, float learning_rate) {
//     for (int i = 0; i < sample_count; i++) {
//             forward(features[i][0], features[i][1], features[i][2], features[i][3]);
//             backpropagate(mlp, features[i], &labels[i], learning_rate);
//         }
// }