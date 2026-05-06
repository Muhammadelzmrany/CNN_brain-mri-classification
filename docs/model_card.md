# Model Card

## Overview

This project uses a convolutional neural network checkpoint named `cnn_improved_brain_tumor.pth` to classify uploaded brain MRI images in a Streamlit interface.

## Intended Use

The model is intended for educational demonstrations, portfolio presentation, and experimentation with medical image classification workflows.

It is not intended for clinical diagnosis, triage, treatment decisions, or any real-world medical workflow.

## Input

- Image format: JPG, JPEG, or PNG
- Color mode: converted to RGB
- Resize: 128 x 128 pixels

## Output Classes

- Glioma
- Meningioma
- No tumor
- Pituitary

## Architecture Summary

- Four convolutional blocks
- Batch normalization after each convolution
- ReLU activations
- Max pooling after each convolutional block
- Fully connected classifier with dropout

Checkpoint metadata:

- `class_names`
- `image_size`
- `model_state_dict`

## Limitations

- No public validation metrics are included in this repository.
- Performance depends on the dataset, preprocessing, image quality, and training procedure used to create the checkpoint.
- MRI interpretation is a medical task and requires specialist review.

## Responsible Use

Display outputs as model predictions only. Do not present them as confirmed diagnoses.

