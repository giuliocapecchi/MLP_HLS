#ifndef MLP_H
#define MLP_H

#define MAX_LAYERS 5               // numero massimo di layer
#define MAX_NEURONS 128            // numero massimo di neuroni per layer
#define MAX_SAMPLES 1000           // numero massimo di campioni
#define MAX_FEATURES 4             // numero massimo di features per campione
#define LEARNING_RATE 0.01
 

#include "utils.h"

/*------------------------ Strutture dati ------------------------*/

typedef struct {
    int input_size;              // numero di neuroni in ingresso al layer
    int output_size;             // numero di neuroni in uscita dal layer
    float weights[MAX_NEURONS][MAX_NEURONS];  // pesi del layer (max neuroni)
    float biases[MAX_NEURONS];   // bias del layer (max neuroni)
    float output[MAX_NEURONS];   // output del layer (max neuroni)
    float errors[MAX_NEURONS];   // errori del layer (max neuroni)
    int activation_function;     // identificatore della funzione di attivazione
} Layer;

typedef struct {
    int num_layers;              // numero di layer nel MLP
    Layer layers[MAX_LAYERS];    // array dei layer
    int loss_function;           // funzione di perdita
} MLP;

/*------------------------ Funzioni per la gestione del MLP ------------------------*/

MLP* create_mlp(int input_neurons, int num_layers, int *number_of_neurons, int *activations, int loss_function);

int forward(MLP *mlp, float *input, int input_size);

void calculate_output_error(MLP *mlp, float *predicted, float *true_value, int loss_function);

void backpropagate(MLP *mlp, float *input, float *true_value, float learning_rate);

void train(MLP *mlp, float features[MAX_SAMPLES][MAX_FEATURES], float labels[MAX_SAMPLES], int sample_count, float learning_rate);

#endif // MLP_H