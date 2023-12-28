# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 17:11:02 2023
@author: Marwa Kechaou

Solution to the problem:
https://baylor.kattis.com/courses/CSI5325/20s/assignments/kwne7w/problems/baylor.perceptron
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Perceptron Learning Algorithm')
# Benchmark specific args
parser.add_argument('--random_seed', default=0, type=int,help='random seed applied to ensure results reproducibility')
parser.add_argument('--max_iter', default=1000, type=int,help='Maximum number of iterations to run PLA')
parser.add_argument('--learning_rate', default=1, type=float,help='step to update model weights')
parser.add_argument('--training_dataset_path', default="./data/simple.in", type=str,help='path of training dataset')
parser.add_argument('--testing_dataset_path', default='./data/simple.ans', type=str,help='path of testing dataset')    


# Read data following the described structure
def extract_data(file_path):
    i = 0
    X =[]
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            line_content = line.strip()
            if i != 0:
                line_content = line_content.split(" ") 
                X.append(list(map(float, line_content[:-1])))
                y.append(int(line_content[-1]))
            i+=1
    return np.array(X),np.array(y)

# Implement the Perceptron Learning Algorithm
def PLA(X, y, max_iter, learning_rate):
    """
    Parameters:
    - X: Input features
    - y: labels
    - max_iter: Maximum number of iterations (default is set to 1000). 
    The stopping condition could be that there are no more misclassified points, 
    indicating that the algorithm has converged and found a solution. 
    The max_iter parameter is a safety measure to avoid infinite loops in case the 
    data is not linearly separable.

    Returns:
    - w: Learned weights
    """
    X = np.column_stack((np.ones(X.shape[0]), X))
    w = np.zeros(X.shape[1])

    for i in range(max_iter):
        misclassified_points = []
        
        for j in range(X.shape[0]):
            if np.sign(np.dot(w, X[j, :])) != y[j]:
                misclassified_points.append(j)

        if not misclassified_points:
            break

        random_misclassified_point = np.random.choice(misclassified_points)

        # Update weights with a learning rate
        w = w + learning_rate * y[random_misclassified_point] * X[random_misclassified_point, :]
    return w

def Perceptron_evaluation(w, X_test, y_test):
    # Add a column for the bias term (w0) to the test features
    X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

    predictions = np.sign(np.dot(X_test, w))
    accuracy = 100*np.mean(predictions == y_test)
    return accuracy
    
def decision_boundary(X_test, y_test, w):
    # Plot test data points
    plt.scatter(X_test[:, 0][y_test==-1], X_test[:, 1][y_test==-1], c=y_test[y_test==-1], cmap=plt.cm.Paired, marker='s', edgecolors='k', label='Negative Test Data')
    plt.scatter(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], c=y_test[y_test==1], cmap=plt.cm.Paired, marker='o', edgecolors='r', label='Positive Test Data')

    # Plot decision boundary
    x_min, x_max = np.min(X_test[:, 0]) - 1, np.max(X_test[:, 0]) + 1
    y_min, y_max = np.min(X_test[:, 1]) - 1, np.max(X_test[:, 1]) + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = np.dot(np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()], w)
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0])
    plt.title('Test Data and Perceptron Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def main():
    args = parser.parse_args()
    # Set the random seed for reproducibility
    np.random.seed(args.random_seed)
    # Get the  training and testing data
    X_train, y_train = extract_data(args.training_dataset_path)
    X_test, y_test = extract_data(args.testing_dataset_path)
    max_iter = args.max_iter
    learning_rate = args.learning_rate
    # Apply Perceptron Learning Algorithm
    w = PLA(X_train, y_train,max_iter,learning_rate)
    # Print and plot results
    print("Learned weights :",",".join(map(str, w)))
    print("training accuracy : ",Perceptron_evaluation(w,X_train, y_train),"%")
    print("testing accuracy : ",Perceptron_evaluation(w,X_test, y_test),"%")
    decision_boundary(X_test, y_test, w)

if __name__ == "__main__":
    main()
