# Diabetic Retinopathy Detector

## Project Overview

Diabetic Retinopathy Detector is a web-based application that uses a Convolutional Neural Network (CNN) to detect diabetic retinopathy from retinal fundus images.

The application allows users to upload retinal images and receive a prediction: No DR or DR Detected, along with a confidence score.

## Problem Statement

Diabetic retinopathy is an eye condition associated with diabetes. This project explores how deep learning can help classify retinal images and support early screening research.

## Objectives

- Develop a CNN-based image classification model.
- Classify retinal images into No DR and DR Detected.
- Build a web interface using Flask.
- Containerize and deploy the application using Podman.
- Demonstrate networking, port mapping and persistent storage.

## Technologies Used

- Python
- TensorFlow and Keras
- Flask
- NumPy
- Pillow
- HTML and CSS
- Podman
- Git and GitHub
- Docker Hub

## Dataset

The project uses the APTOS 2019 Blindness Detection dataset from Kaggle. The original dataset contains five diabetic retinopathy severity grades. This project converts them into two classes:

- 0: No DR
- 1: DR Detected

The dataset is not included in this GitHub repository.

## Model Architecture

The CNN model uses convolutional layers, max pooling, flattening, dense layers and dropout, followed by a sigmoid output layer for binary classification.

Input image size: 224 x 224 x 3

## Model Performance

- Validation accuracy: 95.81%
- Test accuracy: 90.55%%
- Test dataset: 550 images

## Application Features

- Upload retinal fundus images.
- Predict No DR or DR Detected.
- Display prediction confidence.
- View model performance graphs.
- View the model dashboard.

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Run Using Podman

Build the image:

```bash
podman build -t diabetic-retinopathy-detector:1.0 .
```

Create a network and volume:

```bash
podman network create diabetic-network
podman volume create diabetic-uploads
```

Run the container:

```bash
podman run -d --name diabetic-retinopathy-container --network diabetic-network -p 5000:5000 -v diabetic-uploads:/app/static/uploads docker.io/tanmai123/diabetic-retinopathy-detector:1.0
```

Open http://localhost:5000 in your browser.

## Docker Hub

https://hub.docker.com/r/tanmai123/diabetic-retinopathy-detector

## GitHub

https://github.com/KotaTanmai/Diabetic-Retinopathy-Detector

## Disclaimer

This project is developed for educational purposes only. It is not intended to provide medical diagnosis or replace professional medical advice.

