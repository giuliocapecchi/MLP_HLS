#include <stdio.h>
#include <stdlib.h>
#include "ConvNet.h"

#define IMAGE_SIZE 28*28 // MNIST images are 28x28 pixels
#define INPUT_FILE_PATH "./PyTorch/input_image.txt"

// Function to read a single MNIST image from a file
void read_mnist_image(const char *path, float image[IMAGE_SIZE]) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    unsigned char buffer[IMAGE_SIZE];
    fread(buffer, sizeof(unsigned char), IMAGE_SIZE, file);
    fclose(file);

    for (int i = 0; i < IMAGE_SIZE; i++) {
        image[i] = buffer[i] / 255.0; // Normalize pixel values to [0, 1]
    }
}

void read_input_image(const char *file_path, float input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS], int *label) {
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Failed to open input file");
        exit(EXIT_FAILURE);
    }

    // Read label
    fscanf(file, "Label: %d\n", label);

    // Read image values
    for (int h = 0; h < INPUT_HEIGHT; h++) {
        for (int w = 0; w < INPUT_WIDTH; w++) {
            fscanf(file, "{%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f},",
                   &input[h][w][0], &input[h][w][1], &input[h][w][2], &input[h][w][3], &input[h][w][4], &input[h][w][5], 
                   &input[h][w][6], &input[h][w][7], &input[h][w][8], &input[h][w][9], &input[h][w][10], &input[h][w][11], 
                   &input[h][w][12], &input[h][w][13], &input[h][w][14], &input[h][w][15], &input[h][w][16], &input[h][w][17], 
                   &input[h][w][18], &input[h][w][19], &input[h][w][20], &input[h][w][21], &input[h][w][22], &input[h][w][23], 
                   &input[h][w][24], &input[h][w][25], &input[h][w][26], &input[h][w][27]);
        }
    }

    fclose(file);
}

int main() {
    float input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS];
    float output[NUM_CLASSES];
    int label;

    read_input_image(INPUT_FILE_PATH, input, &label);

    int results = forward(input, output);

    if (results != 0) {
        printf("Error during forward pass\n");
        return 1;
    }

    printf("Predicted output:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("Class %d: %f\n", i, output[i]);
    }

    printf("True label: %d\n", label);

    return 0;
}
