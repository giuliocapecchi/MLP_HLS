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

    // Leggi la label
    if (fscanf(file, "Label: %d\n", label) != 1) {
        perror("Failed to read label");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Leggi i valori dell'immagine
    for (int h = 0; h < INPUT_HEIGHT; h++) {
        // Leggi la parentesi graffa di apertura
        char c;
        while ((c = fgetc(file)) != EOF && (c == ' ' || c == '\n')); // Salta spazi e newline
        if (c != '{') {
            perror("Failed to read opening brace");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Leggi i valori della riga
        for (int w = 0; w < INPUT_WIDTH; w++) {
            if (fscanf(file, "%f", &input[h][w][0]) != 1) {
                printf("Failed to read image value at (%d, %d)\n", h, w);
                fclose(file);
                exit(EXIT_FAILURE);
            }

            // Leggi la virgola se non è l'ultimo valore
            if (w < INPUT_WIDTH - 1) {
                if (fscanf(file, ",") != 0) {
                    perror("Failed to read comma between values");
                    fclose(file);
                    exit(EXIT_FAILURE);
                }
            }
        }

        // Leggi la parentesi graffa di chiusura
        while ((c = fgetc(file)) != EOF && (c == ' ' || c == '\n' || c == ',')); // Salta spazi, newline e virgola
        if (c != '}') {
            perror("Failed to read closing brace");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Leggi la virgola dopo la parentesi graffa se non è l'ultima riga
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

    // printf("Input image:\n");
    // for (int h = 0; h < INPUT_HEIGHT; h++) {
    //     for (int w = 0; w < INPUT_WIDTH; w++) {
    //         printf("%f ", input[h][w][0]);
    //     }
    //     printf("\n");
    // }

    // TODO : la rete poi ritornerà solo il max. Per ora lo lascio così per debug
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
