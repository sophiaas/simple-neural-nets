import numpy as np


def conv2d(X, W, stride, pad):
    kernel_height, kernel_width, in_channels, out_channels = W.shape
    n_examples, in_rows, in_cols, in_channels = X.shape
    kernel_shape = (kernel_height, kernel_width)

    X_col, split_shape, X_pad_shape, p = im2col(X, kernel_shape, stride, pad)

    out_rows = int((in_rows + p[0] + p[1] - kernel_height) / stride + 1)
    out_cols = int((in_cols + p[2] + p[3] - kernel_width) / stride + 1)

    # print('kernel_heigh: {}'.format(kernel_height))
    # print('kernel_width: {}'.format(kernel_width))
    # print('in_channels: {}'.format(in_channels))
    # print('out_channels: {}'.format(out_channels))
    # print('n_examples: {}'.format(n_examples))
    # print('in_rows: {}'.format(in_rows))
    # print('in_cols: {}'.format(in_cols))
    # print('new w shape: {}'.format((np.prod(kernel_shape)*in_channels, -1)))

    W_col = W.reshape((np.prod(kernel_shape) * in_channels, -1))
    # W_col = W.transpose(3, 2, 0, 1).reshape(out_channels, -1)
    # print('W col: {}'.format(W_col.shape))
    Z = (
        (X_col @ W_col)
        .reshape(out_channels, out_rows, out_cols, n_examples)
        .transpose(3, 1, 2, 0)
    )
    # print('Z shape: {}'.format(Z.shape))
    return Z, X_col, split_shape, X_pad_shape, p


def im2col(X, kernel_shape, stride=1, pad=0):
    X_pad, p = pad2d(X, pad, kernel_shape, stride)
    # print('X_pad shape: {}'.format(X_pad.shape))
    # print('p shape: {}'.format(p))

    X_split, split_shape = split_windows(X_pad, kernel_shape, stride)
    X_ravel = X_split.reshape(np.prod(X_split.shape[:3]), -1)
    # print('x ravel shape: {}'.format(X_ravel.shape))
    return X_ravel, split_shape, X_pad.shape, p


def col2im(X_col, X_shape, split_shape, kernel_shape, p):
    # print('COL2IM')
    X_split = np.reshape(X_col, split_shape)
    kernel_height, kernel_width = kernel_shape
    # print('reconstructed X_split shape:{}'.format(X_split.shape))
    X_ = np.zeros((X_shape))
    for i in range(X_split.shape[1]):
        for j in range(X_split.shape[2]):
            X_[:, i : i + kernel_height, j : j + kernel_width, :] += X_split[
                :, i, j, :, :, :
            ]

    unpad0, unpad1, unpad2, unpad3 = p
    unpad1 = -unpad1 if unpad1 != 0 else None
    unpad3 = -unpad3 if unpad3 != 0 else None
    X_ = X_[:, unpad0:unpad1, unpad2:unpad3, :]
    # print('FINAL X_ shape: {}'.format(X_.shape))
    return X_


# def col2im(X_col, X_shape, W_shape, stride, pad):
#     p = pad
#     n_examples, in_rows, in_cols, in_channels = X_shape
#     kernel_height, kernel_width, n_in, n_out = W_shape
#     kernel_shape = (kernel_height, kernel_width)
#
#     X_pad = np.zeros((n_examples, n_in, in_rows+p[0]+p[1], in_cols+p[2]+p[3]))
#     X_ = X.transpose(0, 3, 1, 2)
#     i, j, k = im2col_indices(X, kernel_shape, stride, pad)
#
#     X_col_reshaped = X_col.reshape(n_in * kernel_height * kernel_width, -1, n_examples)
#     X_col_reshaped = X_col_reshaped.transpose(2, 0, 1)
#     np.add.at(X_pad, (slice(None), k, i, j), X_col_reshaped)
#
#     p1 = None if p[1] == 0 else -p[1]
#     p3 = None if p[3] == 0 else -p[3]
#
#     return X_pad[:, :, p[0]:p1, p[2]:p3]


def split_windows(X, kernel_shape, stride=1):
    # print('x inside split windows shape: {}'.format(X.shape))
    n_examples, in_rows, in_cols, in_channels = X.shape
    kernel_shape = (n_examples,) + kernel_shape + (in_channels,)
    # print('kernel_shape {}'.format(kernel_shape))
    slices = slice(None, None, stride)
    window_strides = np.array(X.strides)

    indexing_strides = X[slices].strides

    kernel_idx_shape = (
        (np.array(X.shape) - np.array(kernel_shape)) // np.array(stride)
    ) + 1

    new_shape = tuple(list(kernel_idx_shape) + list(kernel_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    X_ = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=strides)
    # print('split X shape: {}'.format(X_.shape))

    X_ = np.squeeze(X_, axis=(0, 3))

    # print('split X shape: {}'.format(X_.shape))
    X_ = X_.transpose(2, 0, 1, 3, 4, 5)
    # print('split X shape: {}'.format(X_.shape))

    return X_, X_.shape


def pad2d(X, pad, kernel_shape=None, stride=None):

    p = pad
    if isinstance(p, int):
        p = (p, p, p, p)

    if isinstance(p, tuple):
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    if p == "same" and kernel_shape and stride is not None:
        p = calc_pad_dims_2D(X.shape, X.shape[1:3], kernel_shape, stride)
        X_pad, p = pad2d(X, p, kernel_shape, stride)
    return X_pad, p


def pool2d(X, kernel_shape, stride, pad, mode="max"):
    kernel_height, kernel_width, in_channels, out_channels = W.shape
    n_examples, in_rows, in_cols, in_channels = X.shape
    kernel_shape = (kernel_height, kernel_width)

    X_col, split_shape, X_pad_shape, p = im2col(X, kernel_shape, stride, pad)

    # print('!!!!!!!!!!!!!!pool X_col shape {}'.format(X_col.shape))

    if mode == "max":
        pool_fn = np.max
    elif mode == "average":
        pool_fn = np.mean

    X_pool = np.zeros((n_examples, out_rows, out_cols, in_channels))
    for m in range(n_ex):
        for i in range(out_rows):
            for j in range(out_cols):
                for c in range(in_channels):
                    # calculate window boundaries, incorporating stride
                    i0, i1 = i * stride, (i * stride) + kernel_height
                    j0, j1 = j * stride, (j * stride) + kernel_width

                    xi = X_pad[m, i0:i1, j0:j1, c]
                    X_pool[m, i, j, c] = pool_fn(xi)

    return X_pool


# def pool2d(X, kernel_shape, stride, pad, mode='max'):
#     n_examples, in_rows, in_cols, in_channels = X.shape
#     kernel_height, kernel_width = kernel_shape
#     X_pad, p = pad2d(X, p, kernel_shape, stride)
#
#     out_rows = int((in_rows + p[0] + p[1] - kernel_height) / stride + 1)
#     out_cols = int((in_cols + p[2] + p[3] - kernel_width) / stride + 1)
#
#     if mode == 'max':
#         pool_fn = np.max
#     elif mode == 'average':
#         pool_fn = np.mean
#
#     X_pool = np.zeros((n_examples, out_rows, out_cols, in_channels))
#     for m in range(n_ex):
#         for i in range(out_rows):
#             for j in range(out_cols):
#                 for c in range(in_channels):
#                     # calculate window boundaries, incorporating stride
#                     i0, i1 = i * stride, (i * stride) + kernel_height
#                     j0, j1 = j * stride, (j * stride) + kernel_width
#
#                     xi = X_pad[m, i0:i1, j0:j1, c]
#                     X_pool[m, i, j, c] = pool_fn(xi)
#
#     return X_pool


def calc_pad_dims_2D(X_shape, out_dim, kernel_shape, stride):
    """
    Compute the padding necessary to ensure that convolving `X` with a 2D kernel
    of shape `kernel_shape` and stride `stride` produces outputs with dimension
    `out_dim`.
    Parameters
    ----------
    X_shape : tuple of `(n_ex, in_rows, in_cols, in_ch)`
        Dimensions of the input volume. Padding is applied to `in_rows` and
        `in_cols`.
    out_dim : tuple of `(out_rows, out_cols)`
        The desired dimension of an output example after applying the
        convolution.
    kernel_shape : 2-tuple
        The dimension of the 2D convolution kernel.
    stride : int
        The stride for the convolution kernel.
    Returns
    -------
    padding_dims : 4-tuple
        Padding dims for `X`. Organized as (left, right, up, down)
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(out_dim, tuple):
        raise ValueError("`out_dim` must be of type tuple")

    if not isinstance(kernel_shape, tuple):
        raise ValueError("`kernel_shape` must be of type tuple")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    n_examples, in_rows, in_cols, in_channels = X_shape

    fr, fc = kernel_shape
    out_rows, out_cols = out_dim

    pr = int((stride * (out_rows - 1) + fr - in_rows) / 2)
    pc = int((stride * (out_cols - 1) + fc - in_cols) / 2)

    out_rows1 = int(1 + (in_rows + 2 * pr - fr) / stride)
    out_cols1 = int(1 + (in_cols + 2 * pc - fc) / stride)

    # add asymmetric padding pixels to right / bottom
    pr1, pr2 = pr, pr
    if out_rows1 == out_rows - 1:
        pr1, pr2 = pr, pr + 1
    elif out_rows1 != out_rows:
        raise AssertionError

    pc1, pc2 = pc, pc
    if out_cols1 == out_cols - 1:
        pc1, pc2 = pc, pc + 1
    elif out_cols1 != out_cols:
        raise AssertionError

    return (pr1, pr2, pc1, pc2)
