#ifndef CONVNET_H
#define CONVNET_H

#define INPUT_HEIGHT 28            // Input height
#define INPUT_WIDTH 28             // Input width
#define INPUT_CHANNELS 1           // Input channels
#define CONV1_OUTPUT_CHANNELS 3    // Number of filters in the first convolutional layer
#define POOL_SIZE 2                // Kernel size for MaxPooling
#define POOL_STRIDE 2              // Stride for MaxPooling
#define FC1_INPUT_SIZE (CONV1_OUTPUT_CHANNELS * (INPUT_HEIGHT / POOL_SIZE) * (INPUT_WIDTH / POOL_SIZE)) // Input size for the fully connected layer
#define NUM_CLASSES 10             // Number of classes (final output)

/*------------------------ Data Structures ------------------------*/

// Structure to represent a convolutional layer
typedef struct {
    float weights[CONV1_OUTPUT_CHANNELS][INPUT_CHANNELS][3][3]; // Filters of the convolutional layer 
    float biases[CONV1_OUTPUT_CHANNELS];                        // Biases for the filters
} ConvLayer;

// Structure to represent a fully connected layer
typedef struct {
    float weights[NUM_CLASSES][FC1_INPUT_SIZE]; // Weights of the fully connected layer
    float biases[NUM_CLASSES];                 // Biases of the fully connected layer
} FullyConnectedLayer;

// General structure of the network
typedef struct {
    ConvLayer conv1;               // First convolutional layer
    FullyConnectedLayer fc1;       // Fully connected layer
} ConvNet;

/*-------------------------- Functions ---------------------------*/

int forward(float input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS], float output[NUM_CLASSES]);

#endif // CONVNET_H
