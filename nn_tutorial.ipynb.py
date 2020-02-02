# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sys
from importlib import reload
from matplotlib import pyplot as plt
import torch
import nn_tutorial as nt

_ = reload(nt)

(x_train, y_train, x_valid, y_valid) = nt.get_data()
# -

plt.imshow(nt.x_train[0].reshape((28, 28)), cmap="gray")

x_train.shape, y_train.shape

# +
n_batch = 64  # a batch
x = x_train[0:n_batch]
y = y_train[0:n_batch]

yp.shape, yp[0]
# -

model = nt.NeuralNetwork()
yp = model.predict(x)
model.loss_fn(yp, y), model.accuracy(yp, y)

model.fit(x_train, y_train)

yp = model.predict(x)
model.loss_fn(yp, y), model.accuracy(yp, y)
