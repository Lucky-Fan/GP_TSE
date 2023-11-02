import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import pickle
import tensorflow as tf
from scipy.io import loadmat, savemat
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
import gpflow
from gpflow.utilities import print_summary, positive
from tensorflow_probability import bijectors as tfb
import random
import os
import matplotlib
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica", "Arial"], "size": 10})
# Set axis tick inwards
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"

def fill_nan(data):
    """Fill nan values with the nearest non-nan value"""
    ind = np.arange(data.shape[0])
    for i in range(data.shape[1]):
        data[:, i] = np.interp(ind, ind[~np.isnan(data[:, i])], data[~np.isnan(data[:, i]), i])
    return data

def reset_random_seeds(n=0):
    os.environ['PYTHONHASHSEED'] = str(n)
    np.random.seed(n)
    random.seed(n)

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mae(predictions, targets):
    return np.abs(predictions - targets).mean()

def SE(r, variance):
    return variance * tf.exp(-0.5 * r ** 2)


def Matern52(r, variance):
    sqrt5 = np.sqrt(5.0)
    return variance * (1 + sqrt5 * r + 5.0 / 3.0 * r ** 2) * tf.exp(-sqrt5 * r)


def Matern32(r, variance):
    sqrt3 = np.sqrt(3.0)
    return variance * (1 + sqrt3 * r) * tf.exp(-sqrt3 * r)


def Rational_Quadratic(r, variance, alpha):
    return variance * (1 + r ** 2 / (2 * alpha)) ** (-alpha)


class Directional_Kernel(gpflow.kernels.Kernel):
    def __init__(self, kernel, variance=1.0, lengthscales=(1.0, 1.0), theta=0.01, alpha=None):
        super().__init__(name=f"Directional_{kernel.__name__}")
        sigmoid = tfb.Sigmoid(tf.cast(0.0, tf.float64), tf.cast(2*np.pi, tf.float64))
        self.variance = gpflow.Parameter(variance, transform=positive(), dtype=tf.float64)
        self.theta = gpflow.Parameter(theta, transform=sigmoid, dtype=tf.float64)
        self.lengthscale = gpflow.Parameter(lengthscales, transform=positive(), dtype=tf.float64)
        self.kernel = kernel
        if alpha is not None:
            self.alpha = gpflow.Parameter(alpha, transform=positive(), dtype=tf.float64)

    def square_distance(self, X, X2=None):
        if X2 is None:
            X2 = X
        rotation_matrix = tf.stack([tf.cos(self.theta), -tf.sin(self.theta), tf.sin(self.theta), tf.cos(self.theta)])
        rotation_matrix = tf.reshape(rotation_matrix, [2, 2])
        X = tf.matmul(X, rotation_matrix)
        X2 = tf.matmul(X2, rotation_matrix)
        X_scaled = X / self.lengthscale
        X2_scaled = X2 / self.lengthscale
        X2_scaled = tf.transpose(X2_scaled)
        return tf.reduce_sum(X_scaled ** 2, 1, keepdims=True) - 2 * tf.matmul(X_scaled, X2_scaled) + tf.reduce_sum(X2_scaled ** 2, 0, keepdims=True)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        # Only use the first two dimensions
        r2 = self.square_distance(X[:, 0:2], X2[:, 0:2])
        r = tf.sqrt(tf.maximum(r2, 1e-36))
        if self.kernel.__name__ == "Rational_Quadratic":
            return self.kernel(r, self.variance, self.alpha)
        else:
            return self.kernel(r, self.variance)

    def K_diag(self, X):
        return tf.fill(tf.shape(X[:, 0:2])[:-1], tf.squeeze(self.variance))
