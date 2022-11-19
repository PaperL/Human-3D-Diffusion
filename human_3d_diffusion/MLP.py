from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import softmax

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)


class Layer():
    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, node_grad):
        raise NotImplementedError

    def update(self, learning_rate):
        raise NotImplementedError


class Relu():
    def __init__(self):
        self.rec = None

    def forward(self, x):
        self.rec = (x <= 0)
        self.y = x
        self.y[self.rec] = 0
        return self.y

    def backward(self, node_grad):
        ans = node_grad
        ans[self.rec] = 0
        return ans

    def update(self, learning_rate):
        pass


class Softmax_Cross_Entropy():
    def __init__(self):
        self.y = None
        self.x = None

    def forward(self, x):
        self.x = x
        self.y = np.exp(x) / np.sum(np.exp(x))
        return self.y

    def backward(self, label):
        return self.y - label

    def update(self, learning_rate):
        pass


class Linear(Layer):
    def __init__(self, size_in, size_out, with_bias):
        super().__init__()
        self.G_b = None
        self.G_w = None
        self.y = None
        self.x = None
        self.size_in = size_in
        self.size_out = size_out
        self.with_bias = with_bias
        self.W = self.initialize_weight()
        if with_bias:
            self.b = np.zeros(size_out)

    def initialize_weight(self):
        epsilon = np.sqrt(2.0 / (self.size_in + self.size_out))
        return epsilon * (np.random.rand(self.size_in, self.size_out) * 2 - 1)

    def forward(self, x):
        self.x = x
        self.y = np.dot(x, self.W) + self.b
        return self.y

    def backward(self, node_grad):
        self.G_w = np.dot(np.array(self.x).reshape(-1, 1), np.array(node_grad).reshape(1, -1))
        self.G_b = node_grad
        return np.dot(node_grad, self.W.T)

    def update(self, learning_rate):
        self.W -= self.G_w * learning_rate
        self.b -= self.G_b * learning_rate


class MLP():
    def __init__(self, layer_size, with_bias=True, learning_rate=1):
        self.layers = None
        assert len(layer_size) >= 2
        self.layer_size = layer_size
        self.with_bias = with_bias
        self.activation = Relu
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        self.layers = []

        size_in = self.layer_size[0]
        for hu in self.layer_size[1:-1]:
            self.layers.append(Linear(size_in, hu, self.with_bias))
            self.layers.append(self.activation())
            size_in = hu

        self.layers.append(Linear(size_in, self.layer_size[-1], self.with_bias))
        self.layers.append(Softmax_Cross_Entropy())

    def forward(self, x, timesteps, y=None):
        tmp = np.resize(timesteps, x.shape)
        x = x + tmp
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, label):
        node_grad = label
        for layer in reversed(self.layers):
            node_grad = layer.backward(node_grad)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x)

    def loss(self, x, label):
        y = self.forward(x)
        return -np.log(y) @ label
