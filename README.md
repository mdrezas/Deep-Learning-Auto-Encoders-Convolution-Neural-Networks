# Deep-Learning-Auto-Encoders-Convolution-Neural-Networks

Background and Methods:
Fashion-MNIST: Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image associated with a label from 10 classes. Fashion-MNIST is intended to serve as a direct drop-in replacement of the original MNIST dataset for benchmarking machine learning algorithms.[1]

This study proposes machine learning models based on Auto-Encoders and Convolutional Neural Networks with SIGMOID/RELU/SOFTMAX transfer Functions and Binary Cross-Entropy/Categorical Cross-Entropy loss functions to classify Fashion-MNIST data.

MLP Model Evaluation:
One way to measure models' performance is to compare the predictions' error for the actual values with training, test, and validation accuracy and loss. Please refer to the MLP Model Summary Table in the appendix below for details on model performance.
Summary:
After comparing models’ output, it turned out that accuracy, loss, and the score for training, validation, and test data play a crucial role in evaluating the model performance. The goals were to build MLP models to demonstrate the power of the Deep Neural Network in terms of classification high dimensional data. The observations-based models’ outputs are as follows:
Convolutional Neural Networks with categorical cross-entropy and MaxPooling performed well with high accuracy and lower loss rate to classify the Fashion-MNIST data. It seems the model performance is promising in classifying Fashion-MNIST data.

Reference: [1] Xiao, H., Rasul, K. & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms (cite arxiv:1708.07747Comment: Dataset is freely available at https://github.com/zalandoresearch/fashion-mnist Benchmark is available at http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)
