# Brain-Tumor-Detection-Project

## **Overview**

This project aims to develop a deep-learning model for **brain tumor detection** using MRI scans. The model will focus on detecting four types of brain conditions: **glioma**, **healthy**, **meningioma**, and **pituitary**. The goal is to create a robust and efficient neural network that can accurately classify MRI images into these categories.

The approach I am taking is a **"bring your own method"** strategy, where I will implement and modify an existing neural network architecture to improve the detection results. This project is a part of the **Applied Deep Learning course**. The dataset and references used for the project are detailed below.

## **References**

1. **Paper 1**: [Automated Detection of Brain Tumors Using MRI Scans and Deep Learning](https://www.nature.com/articles/s41598-024-52823-9)
2. **Paper 2**: [Deep Learning-Based Brain Tumor Classification Using MRI Scans](https://pmc.ncbi.nlm.nih.gov/articles/PMC10453020/#sec4-cancers-15-04172)

These papers provide insights into state-of-the-art techniques and architectures used for brain tumor detection, including CNN-based approaches and data augmentation techniques to improve classification performance.

## **Dataset**

The dataset I plan to use is sourced from **Kaggle**: [Brain Tumor MRI Scans](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data). It consists of MRI scans categorized into **four labels**:
- **Glioma**
- **Healthy**
- **Meningioma**
- **Pituitary**

The dataset is well-organized and labeled, providing the necessary information to train a supervised learning model.

## **Approach**

The project will focus on implementing a **custom neural network architecture** tailored to the brain tumor detection problem. Based on research from the reference papers, I will explore **convolutional neural networks (CNNs)**, **transfer learning**, and **data augmentation** techniques to improve detection accuracy. The model will be trained on the Kaggle dataset, and I will fine-tune hyperparameters to achieve better results.

## **Work Breakdown Structure**

- **Dataset Collection & Preprocessing (6 hours)**:
  - Download the dataset from Kaggle.
  - Clean, preprocess, and split the data into training, validation, and test sets.

- **Model Design & Development (12 hours)**:
  - Research and design the CNN architecture for tumor detection.
  - Explore transfer learning using pre-trained models (e.g., ResNet or VGG).

- **Model Training & Fine-Tuning (15 hours)**:
  - Train the model on the dataset using different configurations and optimizations.
  - Fine-tune hyperparameters like learning rate, batch size, etc.
  - Implement techniques to mitigate overfitting.

- **Model Evaluation & Testing (8 hours)**:
  - Test the model on the validation set.
  - Evaluate performance using metrics such as accuracy, precision, recall, and F1-score.

- **Application & Visualization Development (10 hours)**:
  - Develop a simple application or visualization to showcase the results.
  - Visualize the MRI scan results and model predictions.

- **Final Report & Presentation (4 hours)**:
  - Write the final report summarizing the approach and results.
  - Prepare a presentation or documentation for the project, including code, results, and figures.

