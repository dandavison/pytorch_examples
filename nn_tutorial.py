import gzip
import math
import pickle
from pathlib import Path

import numpy as np
import requests
import torch


def get_data():
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    return tuple(map(torch.tensor, (x_train, y_train, x_valid, y_valid)))


class NeuralNetwork:
    """
    Multinomial Logistic regression implemented as a neural network.

    In this ML training problem we have n observations x_i, where x_i \in \R^d.

    Each x_i is an image with 784 pixels, so d=784.

    We want to classify each x_i to one of K categories. In this case, K=10 -- the 10 possible
    digits.

    Multinomial Logistic Regression can be thought of as laying down a K-dimensional discrete
    probability distribution surface over each point in R^d. So any observation x has an associated
    output value y, given by the value of the surface above x.

    Variable names:

    x      a single input image vector \in R^{784}
    X      (n x 784) matrix containing n input vectors
    xw     the intermediate output of the linear part of the model:
           xw = x @ weights + bias
    yp     the output of the neural net for one input:
           a vector of multinomial logistic probabilities \in \R^{10}
    yhat   the label prediction derived from the probability vector yp
    y      the true label
    """

    def __init__(self):
        self.Weights = torch.randn(784, 10) / math.sqrt(784)  # "Xavier initialisation"
        self.Weights.requires_grad_()
        self.bias = torch.zeros(10, requires_grad=True)

    def predict(self, X):  # x -> yhat
        """
        This maps observation vectors x in R^{784} to output yp vectors in R^{10}.
        """
        return self.activation(X @ self.Weights + self.bias)

    @staticmethod
    def log_softmax(XW):  # XW -> Yp
        """
        This is basically an extension of the logistic transformation to the case of >2 classes. It
        maps the output of the pure linear model (x @ w + b, a vector in R^{10}) to a
        normalized (sums to 1) output vector in R^{10}. In log space.
        """
        return XW - XW.exp().sum(-1).log().unsqueeze(-1)

    @staticmethod
    def negloglike(Yp, y):  # (n x K) -> (n) -> R
        """
        We can interpret the output of the model for observation i (yp_i) as a vector of
        probabilities, one for each category. So the loss associated with predictions yhat is the
        product of the probabilities of the true category, across all observations. In log space,
        and we want the loss to be large when the probability is small (very negative in log
        space), so minus sign.
        """
        n, = y.shape
        return -Yp[range(n), y].mean()  # why not sum?

    activation = log_softmax
    loss_fn = negloglike

    @staticmethod
    def accuracy(Yp, y):
        yhat = torch.argmax(Yp, dim=1)
        return (yhat == y).float().mean()

    def fit(self, X, y):
        lr = 0.5  # learning rate
        epochs = 2  # how many epochs to train for
        n, d = X.shape
        n_batch = 64

        for epoch in range(epochs):
            for i in range((n - 1) // n_batch + 1):
                start_i = i * n_batch
                end_i = start_i + n_batch
                X_ = X[start_i:end_i]
                y_ = y[start_i:end_i]
                Yp = self.predict(X_)
                loss = self.loss_fn(Yp, y_)

                loss.backward()
                with torch.no_grad():
                    self.Weights -= self.Weights.grad * lr
                    self.bias -= self.bias.grad * lr
                    self.Weights.grad.zero_()
                    self.bias.grad.zero_()
