#ifndef UTILS_H
#define UTILS_H

/*--------Struct for easy backpropagation -----*/

typedef struct {
    float (*function)(float);
    float (*derivative)(float);
} ActivationFunction;

/*----------Activation functions--------------*/

float sigmoid(float x);
float sigmoid_derivative(float x);
float reLu(float x);
float reLu_derivative(float x);
float leakyReLu(float x);
float heavySide(float x); // Aggiunto HeavySide per completezza
float linear(float x);
float linear_derivative();
void softmax(float *output, int size);

/*----------Loss functions--------------*/

typedef enum {
    MEAN_SQUARED_ERROR,
    BINARY_CROSS_ENTROPY
} LossFunctionType;

void print_loss_type(LossFunctionType loss_function);

float meanSquaredError(float *predicted, float *true_value, int size);

float binaryCrossEntropy(float *predicted, float *true_value, int size);

/*----------Additional helper functions (approximation for exp/log)-----------*/

// Funzione di approssimazione esponenziale
float exp_approx(float x);

// Funzione di approssimazione logaritmica
float log_approx(float x);

#endif // UTILS_H
