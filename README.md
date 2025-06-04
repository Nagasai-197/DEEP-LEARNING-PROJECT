# DEEP LEARNING PROJECT  
# IMAGE CLASSIFICATION  


**COMPANY:** CODTECH IT SOLUTIONS  
**NAME:** BATHULA SRI HEMANTH NAGA SAI  
**INTERN ID:** CT06DK242  
**DOMAIN:** DATA SCIENCE  
**DURATION:** 6 WEEKS  
**MENTOR:** NEELA SANTOSH  

---

# CIFAR-10 Image Classification using PyTorch 🧠📦

This project implements a Convolutional Neural Network (CNN) using **PyTorch** to classify images from the **CIFAR-10** dataset. It trains a model, evaluates its performance, and provides visualizations of the results.

---

## 🎯 Project Motivation

The goal of this project is to demonstrate a basic yet functional deep learning pipeline using PyTorch. It includes:

- Loading and normalizing image data  
- Defining and training a CNN model  
- Plotting training and validation accuracy  
- Visualizing model predictions on test samples

---

## 🧠 Model Architecture

The model is a simple Convolutional Neural Network with:

- **Conv2d (3→32)** → ReLU → MaxPool  
- **Conv2d (32→64)** → ReLU → MaxPool  
- **Conv2d (64→64)** → ReLU  
- Flatten → Fully Connected (64 units) → Output layer (10 units)

**Loss Function:** `CrossEntropyLoss`  
**Optimizer:** `Adam`  
**Epochs:** `10`

---

## 🗃 Dataset: CIFAR-10

The **CIFAR-10** dataset consists of **60,000 color images (32x32)** across 10 classes:

- `airplane`, `automobile`, `bird`, `cat`, `deer`  
- `dog`, `frog`, `horse`, `ship`, `truck`

The dataset is split into 50,000 training and 10,000 test images. It's automatically downloaded via `torchvision.datasets`.

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

---

## 📊 OUTPUT

Below are visual results from the trained model:

![Accuracy and Loss Graph + Predictions](https://github.com/user-attachments/assets/86f5d338-49c0-4a69-aac7-65a16a79f920)
![Accuracy and Loss Graph + Predictions](https://github.com/user-attachments/assets/59b8342b-4c9b-4f6e-a515-a6ef7035cab8)

```

