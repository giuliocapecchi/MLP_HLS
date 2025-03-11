# Vitis HLS
Hardware Level Synthesis (HLS) on FPGA devices for the forward pass of the following neural networks:

- Multi-layer Perceptron (MLP)
- Convolutional Neural Network (ConvNet)

This repository contains the C implementation of the forward pass for both MLP and ConvNet, designed to be synthesized on FPGA devices using the VITIS IDE.

Take a look at the [documentation](documentation/documentation.pdf) for a complete overview of this project.

The networks were trained using `PyTorch`, then their weights were extracted and hard-coded in their respective `C` files. The C implementations can be found inside the `HLS-implementations` folder.

<img src="documentation/assets/vitis-hls.png" alt="Vitis HLS" width="400">


## Prerequisites
- VITIS IDE (can be found [here](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis.html))
- PyTorch for neural network training

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/giuliocapecchi/Vitis-HLS
    ```
2. Open the VITIS IDE.
3. Import the project. Check the documentation for more details.

## Project Description
This project focuses on the synthesis of the forward pass for two types of neural network architectures: MLP and ConvNet, implemented on an FPGA. The network parameters were obtained using Python and the PyTorch library, then hardcoded into C code for hardware synthesis.

## Workflow Overview
For each neural network architecture, a Jupyter Notebook is provided in the `PyTorch` folder. These notebooks were used to construct and train the models using PyTorch. Once trained, the weights and biases were exported and hardcoded into the corresponding C implementation. The C code, compatible with FPGA synthesis tools such as Vitis HLS/Vivado, can be found inside the `HLS-Implementation` folder.

## Results
The results confirm the successful synthesis and implementation of the MLP and ConvNet forward pass on the FPGA. Detailed performance metrics and resource utilization reports are available in the report.