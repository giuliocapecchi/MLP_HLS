#include <stdio.h>
#include "utils.h"
#include "math.h"

/*----------Activation functions--------------*/

// Sigmoid approssimata per HLS
float sigmoid(float x) {
    //#pragma HLS INLINE
    return 1.0 / (1.0 + exp_approx(-x)); // TODO : controlla che vada bene
}

// ReLU
float reLu(float x) {
    //#pragma HLS INLINE
    return x > 0 ? x : 0;
}

// Leaky ReLU
float leakyReLu(float x) {
    //#pragma HLS INLINE
    return x > 0 ? x : 0.01 * x;
}

// HeavySide
float heavySide(float x) {
    //#pragma HLS INLINE
    return x > 0 ? 1.0 : 0.0;
}

// Linear
float linear(float x) {
    //#pragma HLS INLINE
    return x;
}

/*----------Derivative of activation functions--------------*/

// Derivata del Sigmoid
float sigmoid_derivative(float x) {
    //#pragma HLS INLINE
    return sigmoid(x) * (1.0 - sigmoid(x));
}

// Derivata della Linear activation function
float linear_derivative() {
    //#pragma HLS INLINE
    return 1.0;
}

// Derivata della ReLU
float reLu_derivative(float x) {
    //#pragma HLS INLINE
    return x > 0 ? 1.0 : 0.0;
}

/*----------Loss functions--------------*/

// Mean Squared Error
float meanSquaredError(float *predicted, float *true_value, int size) {
    //#pragma HLS INLINE
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE
        float diff = predicted[i] - true_value[i];
        sum += diff * diff;
    }
    return sum / size;
}

// Binary Cross Entropy
float binaryCrossEntropy(float *predicted, float *true_value, int size) {
    //#pragma HLS INLINE
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE
        float pred = predicted[i] > 0 ? predicted[i] : 1e-6; // Evita log(0)
        float t = true_value[i]; 
        sum += -t * log(pred) - (1.0 - t) * log(1.0 - pred); // TODO: forse ci vuole anche qui una approssimazione del log
    }
    return sum / size;
}


// Approssimazione della funzione esponenzialeù
// altrimenti cordic method?
float exp_approx(float x) {
    //#pragma HLS INLINE
    // taylor al quarto ordine, funziona solo per input in intervallo [-1, 1]
    float result = 1.0 + x + (x * x) / 2 + (x * x * x) / 6 + (x * x * x * x) / 24;
    return result;
}

// HLS INLINE
// forza l'inserimento completo del corpo di una funzione o di un ciclo direttamente nel punto in cui è chiamato, eliminando l'overhead associato alla chiamata di funzione o all'allocazione di risorse hardware separate.
// Riduce la latenza perché elimina il ritardo di chiamata della funzione. Può aumentare l'area hardware utilizzata, perché la logica della funzione viene replicata ovunque venga chiamata.
// Tipicamente per funzioni semplici o molto usate, per ridurre la latenza.

// HLS PIPELINE
// La direttiva HLS PIPELINE suddivide un ciclo o una funzione in più fasi (pipeline), permettendo l'esecuzione di più iterazioni o operazioni simultaneamente, aumentando il throughput del design.
// Permette l'elaborazione di nuove iterazioni a ogni ciclo di clock (o a intervalli specificati, chiamati Initiation Interval - II).
// Aumenta la velocità del sistema a scapito del consumo di risorse hardware.
// Opzioni:
// II=N (specifica l'intervallo di avvio delle iterazioni, ad esempio, 1 ciclo di clock tra una e l'altra).
// rewind (riavvia automaticamente il ciclo alla fine).
// Uso tipico
// Nei loop che processano grandi quantità di dati.
// Per aumentare il throughput in operazioni ripetitive.