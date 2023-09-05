# Text Classification using Convolutional Neural Networks (CNNs)

This Markdown file provides an overview of a Python script for text classification using CNNs. The script includes preprocessing steps, data loading, and model training.

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Conclusion](#conclusion)

## Introduction

Text classification is a common natural language processing (NLP) task where text documents are categorized into predefined classes or categories. In this script, we use CNNs to perform text classification. CNNs are commonly used for image processing but can also be applied to text data by treating text as a one-dimensional signal.

## Data Preprocessing

### Email Preprocessing

The script starts by preprocessing email data. It extracts email domains, removes special characters, and tokenizes the text.

### Subject Preprocessing

Subject lines are processed by removing special characters, decontractions, and chunking of named entities using NLTK's NER. Words less than 2 and greater than 15 characters are also removed.

### Text Preprocessing

The main text is processed by removing email addresses, special characters, tags, and decontractions. Named entities are chunked, and digits and underscores are removed. Words less than 2 and greater than 15 characters are also removed, leaving only alphabetic characters.

## Model Architecture

### Model 1: Using Word Embeddings

- Tokenization: Text is tokenized using the Keras `Tokenizer` class.
- Embedding Layer: Pretrained GloVe word embeddings are used as input.
- Convolutional Layers: Multiple 1D convolutional layers with different kernel sizes are applied.
- MaxPooling Layers: Max-pooling layers are used for down-sampling.
- Dense Layers: Fully connected layers for classification.
- Callbacks: Early stopping, ModelCheckpoint, and F1 score calculation.

### Model 2: Using Character Embeddings

- Tokenization: Text is tokenized at the character level using Keras `Tokenizer`.
- Embedding Layer: Pretrained GloVe word embeddings are used as input.
- Convolutional Layers: 1D convolutional layers are applied.
- MaxPooling Layers: Max-pooling layers for down-sampling.
- Dense Layers: Fully connected layers for classification.
- Callbacks: Early stopping, ModelCheckpoint, and F1 score calculation.

## Training the Model

The models are trained using the preprocessed data. Training and validation datasets are split using StratifiedShuffleSplit. Model checkpoints are saved, and F1 scores are calculated during training.

## Conclusion

This script demonstrates how to perform text classification using CNNs with different embedding strategies. The choice between word and character embeddings depends on the specific task and dataset. Experiment with hyperparameters and preprocessing techniques to optimize model performance.

Feel free to reach out if you have any questions or need further assistance with this code.
