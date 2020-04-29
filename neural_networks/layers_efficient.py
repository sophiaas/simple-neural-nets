"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from neural_networks.utils.convolution import (
    conv2d,
    im2col,
    col2im,
    pool2d,
    pad2d,
    split_windows,
)

from collections import OrderedDict


"""
TODO:
- Convolutional
"""


def initialize_layer(
    name,
    activation=None,
    weight_init=None,
    n_out=None,
    kernel_shape=None,
    stride=None,
    pad=None,
    pool_mode=None,
    keep_dim="first",
    # conv_dimension=None,
    gate_activation=None,
    backprop_truncation=None,
    threshold=None,
    network_topology=None,
    max_degree=None,
    n_clusters=None,
    synchronous=None,
    weight_init_mode=None,
):

    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out,
            activation=activation,
            weight_init=weight_init,
            weight_init_mode=weight_init_mode,
        )

    elif name == "elman":
        return Elman(
            n_out=n_out,
            activation=activation,
            weight_init=weight_init,
            backprop_truncation=backprop_truncation,
            weight_init_mode=weight_init_mode,
        )

    elif name == "lstm":
        return LSTM(
            n_out=n_out,
            activation=activation,
            gate_activation=gate_activation,
            weight_init=weight_init,
            backprop_truncation=backprop_truncation,
            weight_init_mode=weight_init_mode,
        )

    elif name == "convolutional":
        return Convolutional2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            # conv_dimension=conv_dimension,
            weight_init=weight_init,
            weight_init_mode=weight_init_mode,
        )

    elif name == "pool2d":
        return Pool2D(
            kernel_shape=kernel_shape, mode=pool_mode, stride=stride, pad=pad
        )

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class Layer(ABC):
    def __init__(self):
        self.activation = None
        self.optimizer = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z):
        pass

    def clear_gradients(self):
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def _get_parameters(self):
        return [b for a, b in self.parameters.items()]

    def _get_cache(self):
        return [b for a, b in self.cache.items()]

    def _get_gradients(self):
        return [b for a, b in self.gradients.items()]


class FullyConnected(Layer):
    def __init__(
        self, n_out, activation, weight_init="xavier_uniform", weight_init_mode=None
    ):

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(
            weight_init, activation=activation, mode=weight_init_mode
        )

    def _init_parameters(self, X_shape):
        self.n_in = X_shape[1]

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))
        # b = self.init_weights((1, self.n_out))
        # TODO: add back in the option to initialize bias

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": [], "X": []})
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})

    def forward(self, X):
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = X @ W + b
        out = self.activation(Z)

        self.cache["Z"] = Z
        self.cache["X"] = X

        return out

    def backward(self, dLdY):
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = self.cache["Z"]
        X = self.cache["X"]

        dZ = self.activation.backward(Z, dLdY)
        dX = dZ @ W.T
        dW = X.T @ dZ
        dB = dZ.sum(axis=0, keepdims=True)

        self.gradients["W"] = dW
        self.gradients["b"] = dB

        return dX


class Elman(Layer):
    def __init__(
        self,
        n_out,
        activation="tanh",
        weight_init="xavier_uniform",
        backprop_truncation=None,
        weight_init_mode=None,
    ):

        """
        Elman recurrent layer.
        """
        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.backprop_truncation = backprop_truncation
        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(
            weight_init, activation=activation, mode=weight_init_mode
        )

    def _init_parameters(self, X_shape):
        self.n_in = X_shape[1]

        W = self.init_weights((self.n_in, self.n_out))
        U = self.init_weights((self.n_out, self.n_out))
        bx = self.init_weights((1, self.n_out))
        bs = self.init_weights((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "U": U, "bx": bx, "bs": bs})
        self.cache = OrderedDict({"Z": [], "s": [], "X": []})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.parameters.items()}
        )

    def forward_step(self, X):
        W = self.parameters["W"]
        U = self.parameters["U"]
        bx = self.parameters["bx"]
        bs = self.parameters["bs"]
        s = self.cache["s"]

        if len(self.cache["s"]) == 0:
            s0 = np.zeros((self.n_in, self.n_out))
            s = [s0]

        Z = X @ W + bx + s[-1] @ U + bs
        out = self.activation(Z)

        self.cache["Z"].append(Z)
        self.cache["s"].append(out)
        self.cache["X"].append(X)

        return out

    def forward(self, X):
        if self.n_in is None:
            self._init_parameters(X.shape[:2])

        Y = []
        for t in range(X.shape[2]):
            y = self.forward_step(X[:, :, t])
            Y.append(y)

        return Y[-1]

    def backward(self, dLdY):
        W = self.parameters["W"]
        U = self.parameters["U"]

        Xs = self.cache["X"]
        Zs = self.cache["Z"]
        Ys = self.cache["s"]
        dY_history = np.zeros_like(Ys[0])

        dLdX = []

        for t in reversed(range(len(Ys))):
            dY = dLdY + dY_history
            dZ = self.activation.backward(Zs[t], dY)
            dX = dZ @ W.T

            self.gradients["W"] += Xs[t].T @ dZ
            self.gradients["U"] += Ys[t].T @ dZ
            self.gradients["bx"] += dZ.sum(axis=0, keepdims=True)
            self.gradients["bs"] += dZ.sum(axis=0, keepdims=True)

            dY_history = dZ @ U.T

            dLdX = [dX] + dLdX

        return dLdX


class LSTM(Layer):
    def __init__(
        self,
        n_out,
        activation="tanh",
        gate_activation="sigmoid",
        weight_init="xavier_uniform",
        backprop_truncation=None,
        weight_init_mode=None,
    ):

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.backprop_truncation = backprop_truncation
        self.activation = initialize_activation(activation)
        self.gate_activation = initialize_activation(activation)
        self.init_weights = initialize_weights(
            weight_init, activation=activation, mode=weight_init_mode
        )

    def _init_parameters(self, X_shape):
        self.n_in = X_shape[1]

        Wf, Wu, Wo, Wc = [
            self.init_weights((self.n_in + self.n_out, self.n_out))
        ] * 4
        bf, bu, bo, bc = [self.init_weights((1, self.n_out))] * 4

        self.parameters = OrderedDict(
            {
                "Wf": Wf,
                "Wu": Wu,
                "Wo": Wo,
                "Wc": Wc,
                "bf": bf,
                "bu": bu,
                "bo": bo,
                "bc": bc,
            }
        )
        self.cache = OrderedDict(
            {
                "C_": [],
                "C": [],
                "Gc_": [],
                "Gf_": [],
                "Gf": [],
                "Gu_": [],
                "Gu": [],
                "Go_": [],
                "Go": [],
                "X": [],
                "s": [],
            }
        )
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.parameters.items()}
        )

    def forward_step(self, X):
        Wf, Wu, Wo, Wc, bf, bu, bo, bc = self._get_parameters()
        s = self.cache["s"]
        old_C = self.cache["C"]

        if len(s) == 0:
            s0 = np.zeros((X.shape[0], self.n_out))
            c0 = np.zeros((X.shape[0], self.n_out))
            s = [s0]
            old_C = [c0]

        Z = np.hstack([s[-1], X])

        Gf_ = Z @ Wf + bf
        Gu_ = Z @ Wu + bu
        Go_ = Z @ Wo + bo
        Gc_ = Z @ Wc + bc
        C_ = self.activation(Gc_)
        Gf = self.gate_activation(Gf_)
        Gu = self.gate_activation(Gu_)
        Go = self.gate_activation(Go_)
        C = Gf * old_C[-1] + Gu * C_
        out = Go * self.activation(C)

        self.cache["C"].append(C)
        self.cache["C_"].append(C_)
        self.cache["s"].append(out)
        self.cache["Gc_"].append(Gc_)
        self.cache["Gf"].append(Gf)
        self.cache["Gf_"].append(Gf_)
        self.cache["Gu"].append(Gu)
        self.cache["Gu_"].append(Gu_)
        self.cache["Go"].append(Go)
        self.cache["Go_"].append(Go_)
        self.cache["X"].append(X)

        return out

    def forward(self, X):
        if self.n_in is None:
            self._init_parameters(X.shape[:2])

        Y = []
        for t in range(X.shape[2]):
            y = self.forward_step(X[:, :, t])
            Y.append(y)

        return Y[-1]

    def backward(self, dLdY):
        Wf, Wu, Wo, Wc, bf, bu, bo, bc = self._get_parameters()
        C_, C, Gc_, Gf_, Gf, Go_, Go, Gu_, Gu, X, Y = self._get_cache()

        dY_history = np.zeros_like(Y[0])
        dC_history = np.zeros_like(C[0])

        dLdX = []

        for t in reversed(range(len(Y))):
            dY = dLdY + dY_history
            dC = dC_history + Go[t] * self.activation.backward(C[t], dY)

            dGo = self.activation(C[t]) * self.gate_activation.backward(Go_[t], dY)
            dC_ = Gu[t] * self.activation.backward(Gc_[t], dC)
            dGu = C_[t] * self.gate_activation.backward(Gu_[t], dC)
            dGf = C[t] * self.gate_activation.backward(Gf_[t], dC)

            dZ = dGf @ Wf.T + dGu @ Wu.T + dC_ @ Wc.T + dGo @ Wo.T
            dX = dZ[:, self.n_out :]

            Z = np.hstack([Y[t], X[t]])

            self.gradients["Wf"] += Z.T @ dGf
            self.gradients["Wu"] += Z.T @ dGu
            self.gradients["Wo"] += Z.T @ dGo
            self.gradients["Wc"] += Z.T @ dC_
            self.gradients["bf"] += dGf.sum(axis=0, keepdims=True)
            self.gradients["bu"] += dGu.sum(axis=0, keepdims=True)
            self.gradients["bo"] += dGo.sum(axis=0, keepdims=True)
            self.gradients["bc"] += dC_.sum(axis=0, keepdims=True)

            dY_history = dZ[:, : self.n_out]
            dC_history = Gf * dC

            dLdX = [dX] + dLdX

            return dLdX


class Conv2D(Layer):
    def __init__(
        self,
        n_out,
        kernel_shape,
        activation,
        stride=1,
        pad="same",
        weight_init="xavier_uniform",
        weight_init_mode=None,
    ):

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad
        self.conv = conv2d

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(
            weight_init, activation=activation, mode=weight_init_mode
        )

    def _init_parameters(self, X_shape):
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))
        # b = self.init_weights((1, self.n_out))
        # TODO: add back in the option to initialize bias

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": [], "X": []})
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})

    def forward(self, X):
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]
        # print('W shape: {}'.format(W.shape))
        Z = self.conv(X, W, stride=self.stride, pad=self.pad) + b
        out = self.activation(Z)

        self.cache["Z"] = Z
        self.cache["X"] = X

        return out

    def backward(self, dLdY):
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = self.cache["Z"]
        X = self.cache["X"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        dZ = self.activation.backward(Z, dLdY)

        dZ_col = dZ.transpose(3, 1, 2, 0).reshape(dLdY.shape[-1], -1)
        X_col, p = im2col(X, kernel_shape, self.stride, self.pad)
        W_col = W.transpose(3, 2, 0, 1).reshape(out_channels, -1).T

        dW = (
            (dZ_col @ X_col.T)
            .reshape(out_channels, in_channels, kernel_height, kernel_width)
            .transpose(2, 3, 1, 0)
        )
        dB = dZ_col.sum(axis=1).reshape(1, -1)

        dX_col = W_col @ dZ_col
        dX = col2im(dX_col, X, W.shape, self.stride, p).transpose(0, 2, 3, 1)

        self.gradients["W"] = dW
        self.gradients["b"] = dB

        return dX


class Pool2D(Layer):
    def __init__(self, kernel_shape, mode="max", stride=1, pad=0):

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad
        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X):
        # print('POOL X SHAPE: {}'.format(X.shape))
        s = self.stride
        n_examples, in_rows, in_cols, in_channels = X.shape

        # X = X.reshape(n_examples * in_channels, 1, in_rows, in_cols)
        X_pad, p = pad2d(X, self.pad, self.kernel_shape, self.stride)
        X_split, split_shape = split_windows(X_pad, self.kernel_shape, self.stride)
        # print('POOL Xsplit shape: {}'.format(X_split.shape))

        X_split = X_split.transpose(0, 1, 2, 5, 3, 4)
        # print('POOL Xsplit tranpose shape: {}'.format(X_split.shape))

        X_col = X_split.reshape((np.prod(X_split.shape[:4]), -1))
        # X_col, split_shape, X_pad_shape, p = im2col(X, self.kernel_shape, self.stride, self.pad)

        # print('POOL XCOL: {}'.format(X_col.shape))

        X_pool = self.pool_fn(X_col, axis=1)
        # print('x pool shape: {}'.format(X_pool.shape))
        X_pool_idx = np.argmax(X_col, axis=1)
        mask = np.zeros(X_col.shape, dtype=int)
        # print(mask.shape)
        mask[X_pool_idx] = 1
        # mask = mask.reshape(X_split.shape)
        # print(mask.shape)

        # X_masked = X_split[:, ]

        kernel_height, kernel_width = self.kernel_shape
        out_rows = int((in_rows + p[0] + p[1] - kernel_height) / s + 1)
        out_cols = int((in_cols + p[2] + p[3] - kernel_width) / s + 1)

        X_pool = X_pool.reshape((n_examples, out_rows, out_cols, in_channels))

        self.cache["out_rows"] = out_rows
        self.cache["out_cols"] = out_cols
        self.cache["X_col"] = X_col
        self.cache["X_pool_idx"] = X_pool_idx
        self.cache["X_split_shape"] = X_split.shape
        self.cache["X_pool"] = X_pool
        self.cache["X_shape"] = X.shape
        self.cache["mask"] = mask
        # self.cache['p'] = p

        return X_pool

    def backward(self, dLdY):
        # print('BACKWARDPOOL!!!!')
        s = self.stride
        mask = self.cache["mask"]
        dLdY_ravel = dLdY.reshape(-1, 1)
        # print(dLdY_ravel.shape)
        dLdY = mask * dLdY_ravel

        X_split_shape = self.cache["X_split_shape"]

        dLdY = dLdY.reshape(X_split_shape)
        dLdY = dLdY.transpose(0, 1, 2, 4, 5, 3)

        dX = np.zeros(self.cache["X_shape"])

        # print('mask shape: {}'.format(mask.shape))
        # print('dLdY shape: {}'.format(dLdY.shape))

        # row0 = (i0 * self.stride).reshape(-1, 1)
        # row1 = (row0 + self.kernel_shape[0]).reshape(-1, 1)
        # col0 = (i1 * self.stride).reshape(-1, 1)
        # col1 = (col0 + self.kernel_shape[1]).reshape(-1, 1)

        for i in np.arange(dLdY.shape[1]):
            for j in np.arange(dLdY.shape[2]):
                i0 = i * self.stride
                i1 = i0 + self.kernel_shape[0]
                j0 = j * self.stride
                j1 = j0 + self.kernel_shape[1]
                dX[:, i0:i1, j0:j1, :] += dLdY[:, i, j, :, :, :]
        # print(dX[:,row1:1])

        # dX[:, row0:row1, col0:col1, :] += mask[:, row0:row1, col0:col1, :, :, :]

        return dX

        # X_col = self.cache['X_col']
        # print('pool dldy shape: {}'.format(dLdY.shape))
        # # p = self.cache['p']
        # X_pool = self.cache['X_pool']
        # X_shape = self.cache['X_shape']
        # X_pool_idx = self.cache['X_pool_idx']
        # X_split_shape = self.cache['X_split_shape']
        # print('X_pool shape: {}'.format(X_pool.shape))
        #
        # print('X_pool idx: {}'.format(X_pool_idx[0].shape))
        # print('POOL X shape: {}'.format(X_shape))
        #
        # dX = np.zeros(X_shape)
        #
        # i0 = np.arange(dLdY.shape[1])
        # i1 = np.arange(dLdY.shape[2])
        # row0 = i0 * self.stride
        # row1 = row0 + self.kernel_shape[0]
        # col0 = i1 * self.stride
        # col1 = col0 + self.kernel_shape[1]

        # masks = [np.zeros(self.kernel_size) for x in X_pool_idx]
        #
        # masks = [np.zeros()]
        #
        # dX[:, row0:row1, col0:col1, :] =
        #
        #
        # for i in dLdY.shape[1]:
        #     for j in dLdY.shape[2]:
        #         dX[]
        # pool_shape = self.cache['pool_shape']

        # dLdY_split, split_shape = split_windows(dLdY, self.kernel_shape, self.stride)
        # print('POOL dLdY shape: {}'.format(dLdY_split.shape))

        # dLdY_split = dLdY_split.transpose(0, 1, 2, 5, 3, 4)
        # print('POOL dLdY tranpose shape: {}'.format(dLdY_split.shape))

        # dLdY_split_ravel= dLdY_split.reshape((np.prod(dLdY_split.shape[:4]), -1))

        # if self.mode == 'max':
        #     idxs = self.arg_pool_fn(X_col, axis=1)
        #     print('idxs shape: {}'.format(idxs.shape))
        #     mask = np.zeros_like(dLdY_split_ravel).astype(bool)
        #     mask[idxs] = True
        #     dX = dLdY_split_ravel * mask
        #
        # elif self.mode == 'average':
        #     dX = np.mean(dLdY, axis=1)

        # dX = dX.reshape(dLdY_split.shape)

    # def backward(self, dLdY):
    #     print('back pool dldy shape: {}'.format(dLdY.shape))
    #     s = self.stride
    #     X = self.cache['X']
    #
    #     # n_examples, in_rows, in_cols, in_channels = X.shape
    #     kernel_height, kernel_width = self.kernel_shape
    #     X_pad, p = pad2d(X, self.pad, self.kernel_shape, s)
    #
    #     out_rows = self.cache['out_rows']
    #     out_cols = self.cache['out_cols']
    #
    #     dX = np.zeros_like(X_pad)
    #
    #     for m in range(n_examples):
    #         for i in range(out_rows):
    #             for j in range(out_cols):
    #                 for c in range(in_channels):
    #                     # calculate window boundaries, incorporating stride
    #                     i0, i1 = i * s, (i * s) + kernel_height
    #                     j0, j1 = j * s, (j * s) + kernel_width
    #
    #                     if self.mode == "max":
    #                         xi = X_pad[m, i0:i1, j0:j1, c]
    #                         x, y = np.argwhere(xi == self.pool_fn(xi))[0]
    #                         mask = np.zeros_like(xi).astype(bool)
    #                         mask[x, y] = True
    #                         dX[m, i0:i1, j0:j1, c] += mask * dLdY[m, i, j, c]
    #
    #                     elif self.mode == "average":
    #                         dX[m, i0:i1, j0:j1, c] += self.pool_fn(dLdY[m, i, j, c])
    #     return dX


class Flatten(Layer):
    def __init__(self, keep_dim="first"):
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X, retain_derived=True):
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY):
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX


class Convolutional2D(Layer):
    def __init__(
        self,
        n_out,
        kernel_shape,
        activation,
        # conv_dimension=2,
        stride=1,
        pad="same",
        weight_init="xavier_uniform",
        weight_init_mode=None,
    ):

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad
        self.conv = conv2d

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(
            weight_init, activation=activation, mode=weight_init_mode
        )

    def _init_parameters(self, X_shape):
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))
        # b = self.init_weights((1, self.n_out))
        # TODO: add back in the option to initialize bias

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict(
            {
                "Z": [],
                "X_shape": [],
                "X_pad_shape": [],
                "X_col": [],
                "split_shape": [],
                "p": [],
            }
        )
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})

    def forward(self, X):
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]
        # print('W shape: {}'.format(W.shape))
        Z, X_col, split_shape, X_pad_shape, p = self.conv(
            X, W, stride=self.stride, pad=self.pad
        )
        Z += b
        out = self.activation(Z)

        self.cache["Z"] = Z
        self.cache["X_shape"] = X.shape
        self.cache["X_pad_shape"] = X_pad_shape
        self.cache["X_col"] = X_col
        self.cache["split_shape"] = split_shape
        self.cache["p"] = p

        return out

    def backward(self, dLdY):
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = self.cache["Z"]
        X_shape = self.cache["X_shape"]
        X_pad_shape = self.cache["X_pad_shape"]
        X_col = self.cache["X_col"]
        split_shape = self.cache["split_shape"]
        p = self.cache["p"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X_shape
        kernel_shape = (kernel_height, kernel_width)

        # print('X_col shape: {}'.format(X_col.shape))

        dZ = self.activation.backward(Z, dLdY)
        # print('dZ shape: {}'.format(dZ.shape))

        dZ_col = dZ.transpose(3, 1, 2, 0).reshape(dLdY.shape[-1], -1)
        # print('dZ col shape: {}'.format(dZ_col.shape))

        W_col = W.transpose(3, 2, 0, 1).reshape(out_channels, -1).T
        # print('W col shape: {}'.format(W_col.shape))

        dW = (
            (dZ_col @ X_col)
            .reshape(out_channels, in_channels, kernel_height, kernel_width)
            .transpose(2, 3, 1, 0)
        )
        # print('dW shape: {}'.format(dW.shape))

        dB = dZ_col.sum(axis=1).reshape(1, -1)
        # print('dB shape: {}'.format(dB.shape))

        dX_col = W_col @ dZ_col
        # print('dX_col shape: {}'.format(dX_col.shape))
        dX = col2im(dX_col, X_pad_shape, split_shape, kernel_shape, p)
        # print('dX shape: {}'.format(dX.shape))

        self.gradients["W"] = dW
        self.gradients["b"] = dB

        return dX
