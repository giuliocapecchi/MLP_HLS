#include <stdio.h>
#include <stdlib.h>

#define IMAGE_SIZE 28*28 // MNIST images are 28x28 pixels

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

int main() {
    float image[IMAGE_SIZE];

    // Read a single MNIST image from file
    const char *path = "./datasets/mnist/single_image.bin";
    read_mnist_image(path, image);

    // Call the forward function with the MNIST image
    int prediction = forward(image);

    // Print the prediction
    printf("Prediction: %d\n", prediction);

    return 0;
}
