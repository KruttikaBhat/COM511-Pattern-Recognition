import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import math
from scipy.stats import multivariate_normal

iris = datasets.load_iris()

print(iris)

X = iris.data[:, :4]
y = iris.target
print(X)
print(y[0:150])
print(type(y))

setosa = X[0:50]
virginica = X[50:100]
versicolor = X[100:150]

setosa_train, setosa_test, setosa_train_labels, setosa_test_labels = train_test_split(setosa, y[0:50], test_size = 0.2)
virginica_train, virginica_test, virginica_train_labels, virginica_test_labels = train_test_split(virginica, y[50:100], test_size = 0.2)
versicolor_train, versicolor_test, versicolor_train_labels, versicolor_test_labels = train_test_split(versicolor, y[100:150], test_size = 0.2)

print(setosa_train,  setosa_test, setosa_train_labels, setosa_test_labels, sep = '\n')

print(virginica_train,  virginica_test, virginica_train_labels, virginica_test_labels, sep = '\n')

print(versicolor_train,  versicolor_test, versicolor_train_labels, versicolor_test_labels, sep = '\n')

def computeMuSigma(x):
    mu = np.mean(x, axis = 0)
    mu = mu.reshape(x.shape[1], 1)
    sigma = np.cov(x.T)
    det_sigma = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    return mu, sigma,  det_sigma, sigma_inv
    
mu1, sigma1, det_sigma1, sigma_inv1 = computeMuSigma(setosa_train)
print(mu1, sigma1, det_sigma1, sigma_inv1, sep = '\n\n')

mu2, sigma2, det_sigma2, sigma_inv2 = computeMuSigma(virginica_train)
print(mu2, sigma2, det_sigma2, sigma_inv2, sep = '\n\n')

mu3, sigma3, det_sigma3, sigma_inv3 = computeMuSigma(versicolor_train)
print(mu3, sigma3, det_sigma3, sigma_inv3, sep = '\n\n')


def classify_pi(x):
    P = np.zeros(3);
    
    P1 = multivariate_normal.pdf(x, mean=mu1.reshape(4), cov=sigma1)#computeMulProb(x, mu1,sigma_inverse1, det_sigma1)
    P[0] = P1
    
    P2 = y = multivariate_normal.pdf(x, mean=mu2.reshape(4), cov=sigma2)#computeMulProb(x, mu2,sigma_inverse2, det_sigma1)
    P[1] = P2
    
    P3 = y = multivariate_normal.pdf(x, mean=mu3.reshape(4), cov=sigma3)#computeMulProb(x, mu3,sigma_inverse3, det_sigma1)
    P[2] = P3 
    return (np.argmax(P))
    
setosa_test_predictions = np.zeros(10)
virginica_test_predictions = np.zeros(10)
versicolor_test_predictions = np.zeros(10)

for i in range(setosa_test.shape[0]):
    setosa_test_predictions[i] = classify_pi(setosa_test[i])
    
for i in range(virginica_test.shape[0]):
    virginica_test_predictions[i] = classify_pi(virginica_test[i])
    
for i in range(versicolor_test.shape[0]):
    versicolor_test_predictions[i] = classify_pi(versicolor_test[i])
    
    
print(setosa_test_predictions)
print(virginica_test_predictions)
print(versicolor_test_predictions)


def evaluate_accuracy(predictions, labels):
    return np.sum( np.array(predictions == labels) ) / len(predictions)
    
training_accuracy = np.zeros(4)
training_accuracy[0] = evaluate_accuracy(setosa_test_predictions, setosa_test_labels)
training_accuracy[1] = evaluate_accuracy(virginica_test_predictions, virginica_test_labels)
training_accuracy[2] = evaluate_accuracy(versicolor_test_predictions, versicolor_test_labels)
print(training_accuracy[:3]) 

test_predictions = np.append(np.append(setosa_test_predictions, virginica_test_predictions), versicolor_test_predictions)
print(test_predictions)
test_labels = np.append(np.append(setosa_test_labels, virginica_test_labels), versicolor_test_labels)
training_accuracy[3] = evaluate_accuracy(test_predictions, test_labels)
print(training_accuracy[3])

plt.scatter([1, 2, 3, 4], training_accuracy)
plt.show()
