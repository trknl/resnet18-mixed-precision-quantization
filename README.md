# ResNet-18 Mixed-Precision Quantization

A PyTorch and Brevitas project exploring neural network weight quantization, activation calibration, and mixed-precision configurations for image classification.

## Overview

This project implements a configurable quantization workflow for a ResNet-18 model trained on CIFAR-10.

The workflow includes:

- Batch Normalization fusion into convolution layers
- Per-channel weight quantization using Brevitas
- Configurable weight bit widths for individual convolutional and linear layers
- Activation quantization and calibration
- Accuracy evaluation on the CIFAR-10 test set
- Saving calibrated quantized model checkpoints
- Optional ONNX QCDQ export

## Methodology

The implementation supports uniform and mixed-precision weight configurations. Activation quantization is configured separately.

The experiments aim to investigate the trade-off between model accuracy and reduced weight precision.

Quantized numerical representations do not necessarily imply that inference uses native low-bit hardware instructions. Actual deployment performance depends on the execution backend and supported operators.

## Requirements

- Python
- PyTorch
- torchvision
- Brevitas
- ONNX (for model export)

See `requirements.txt` for the package dependencies.

## Usage

1. Install the required dependencies.
2. Provide a compatible trained ResNet-18 checkpoint.
3. Configure the dataset and checkpoint paths.
4. Select the desired per-layer weight bit widths.
5. Run the quantization and evaluation workflow.

The trained checkpoint and dataset are not included in this repository.

## Project Status

This repository is being prepared for reproducible experiments in neural network compression and quantization.

## Future Work

- Automated comparison of uniform and mixed-precision configurations
- Reproducible accuracy and model-size reports
- Inference benchmarking on supported hardware backends

