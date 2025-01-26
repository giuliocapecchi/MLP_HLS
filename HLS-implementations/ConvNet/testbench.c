#include <stdio.h>
#include <stdlib.h>
#include "ConvNet.h"

#define INPUT_FILE_PATH "./input_image.txt"
#define IMAGE_SIZE (INPUT_HEIGHT * INPUT_WIDTH) // Size of the input image



void read_input_image(const char *file_path, float input[INPUT_HEIGHT][INPUT_WIDTH][1], int *label) {
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Failed to open input file");
        exit(EXIT_FAILURE);
    }

    // read the label
    if (fscanf(file, "Label: %d\n", label) != 1) {
        perror("Failed to read label");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // read image data
    for (int h = 0; h < INPUT_HEIGHT; h++) {
        // read opening brace
        char c;
        while ((c = fgetc(file)) != EOF && (c == ' ' || c == '\n')); // skip spaces and newlines
        if (c != '{') {
            perror("Failed to read opening brace");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // read image values
        for (int w = 0; w < INPUT_WIDTH; w++) {
            if (fscanf(file, "%f", &input[h][w][0]) != 1) {
                printf("Failed to read image value at (%d, %d)\n", h, w);
                fclose(file);
                exit(EXIT_FAILURE);
            }

            // read comma if not last value
            if (w < INPUT_WIDTH - 1) {
                if (fscanf(file, ",") != 0) {
                    perror("Failed to read comma between values");
                    fclose(file);
                    exit(EXIT_FAILURE);
                }
            }
        }

        // read closing brace
        while ((c = fgetc(file)) != EOF && (c == ' ' || c == '\n' || c == ',')); // skip spaces, newlines and commas
        if (c != '}') {
            perror("Failed to read closing brace");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // read the comma if not last row
        if (h < INPUT_HEIGHT - 1) {
            if (fscanf(file, ",") != 0) {
                perror("Failed to read comma between rows");
                fclose(file);
                exit(EXIT_FAILURE);
            }
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

     // Find the class with the highest probability
    float max_prob = output[0];
    int predicted_label = 0;
    
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_label = i;
        }
    }
    printf("Predicted label: %d\n", predicted_label);
    printf("True label: %d\n", label);

    return 0;
}
