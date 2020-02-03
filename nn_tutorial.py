import gzip
import math
import pickle
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch import nn


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


class MnistLogistic(nn.Module):
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
        super().__init__()
        self.Weights = nn.Parameter(
            torch.randn(784, 10) / math.sqrt(784)
        )  # "Xavier initialisation" http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, X):  # X -> XW
        """
        This maps observation vectors x in R^{784} to output yp vectors in R^{10}.
        """
        return X @ self.Weights + self.bias

    # cross_entropy combines log softmax activation with negative log likelihood.
    loss_fn = staticmethod(F.cross_entropy)

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
                XW = self(X_)
                loss = self.loss_fn(XW, y_)

                loss.backward()
                with torch.no_grad():
                    for param in self.parameters():
                        param -= param.grad * lr
                    self.zero_grad()
