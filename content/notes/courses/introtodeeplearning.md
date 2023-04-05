---
title: "Intro to Deep Learning"
date: 2023-04-04T12:47:06-04:00
draft: false
showToc: true
math: true
---

# Why Deep Learning and Why Now?
+ Not having a human do dauting tasks
+ We have more data, hardware and software now

# The Perceptron
+ It is a single neuron that takes input of Xn

# Activation Function
+ Introduce non-linearity so that we can better understand the real world

# Output of the perceptron
+ Three steps to get the output: Dot product, bias, and non-linearity

# Empirical Loss
+ Same as loss function, or cost function -> we want to minimize it

# Gradient Descent
1. Intialize weights randomly 
2. Loop until convergence (compute gradient, update weights)
3. Return weights

# Difficulty in optmizing
+ Learning rate can be tricky to set, because we can lose the global minima
+ We can solve it by using adaptive learning rates

# Mini-batches
+ Leads us to faster training and more accurate results

# Stochastic Gradient Descent
1. Intialize weights randomly 
2. Loop until convergence (pick batch of B data points, compute gradient, update weights)
3. Return weights

# Regularization
+ Discorages our model to learn too complex functions, therefore avoiding overfitting
+ One famous example is Dropout
