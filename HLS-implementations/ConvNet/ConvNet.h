#ifndef MLP_H
#define MLP_H

#define MAX_LAYERS 3                // max number of layers
#define MAX_NEURONS 100             // max neurons per layer
# define INPUT_HEIGHT 28            // input height
# define INPUT_WIDTH 28             // input width
# define INPUT_CHANNELS 1           // input channels
# define CONV1_OUTPUT_CHANNELS 32   // output channels of the first convolutional layer
# define CONV2_OUTPUT_CHANNELS 64   // output channels of the second convolutional layer
# define FC_OUTPUT_SIZE 128         // output size of the fully connected layer
# define NUM_CLASSES 10             // number of classes


/*------------------------ Data Structures ------------------------*/

typedef struct {
    float weights[MAX_NEURONS][MAX_NEURONS];  // weights of the layer
    float biases[MAX_NEURONS];   // biases of the layer
    float output[MAX_NEURONS];   // output of the layer
} Layer;

typedef struct {
    int num_layers;              // number of layers
    Layer layers[MAX_LAYERS];    // layers of the MLP (array of layers)
} ConvNet;

/*-------------------------- Functions ---------------------------*/

int forward(float input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS]);

#endif // MLP_H
