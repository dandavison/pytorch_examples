import gzip
import math
import pickle
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


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
        self.linear_layer = nn.Linear(784, 10)

    def forward(self, X):  # X -> XW
        """
        This maps observation vectors x in R^{784} to output yp vectors in R^{10}.
        """
        return self.linear_layer(X)


def accuracy(Yp, y):
    yhat = torch.argmax(Yp, dim=1)
    return (yhat == y).float().mean()


def fit(model, train_dl, valid_dl, epochs, loss_fn, opt):
    for epoch in range(epochs):
        model.train()
        for X_, y_ in train_dl:
            loss = loss_fn(model(X_), y_)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_fn(model(X_), y_) for (X_, y_) in valid_dl) / len(valid_dl)
            valid_accuracy = sum(accuracy(model(X_), y_) for (X_, y_) in valid_dl) / len(valid_dl)

        print("validation:", epoch, valid_loss, valid_accuracy)


def get_data(n_batch):
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((X_train, y_train), (X_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    (X_train, y_train, X_valid, y_valid) = map(torch.tensor, (X_train, y_train, X_valid, y_valid))
    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=n_batch, shuffle=True)
    valid_dl = DataLoader(TensorDataset(X_valid, y_valid), batch_size=n_batch * 2)

    return train_dl, valid_dl


def train_logistic(train_dl, valid_dl):
    model = MnistLogistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.5)
    fit(model, train_dl, valid_dl, epochs=2, loss_fn=loss_fn, opt=opt)
    return model, loss_fn




if __name__ == "__main__":
    train_dl, valid_dl = get_data(n_batch=64)
    train_logistic(train_dl, valid_dl)
    train_cnn(train_dl, valid_dl)
