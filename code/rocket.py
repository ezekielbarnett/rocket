from sklearn.linear_model import RidgeClassifierCV
from numba import njit, prange
import numpy as np


@njit
def generate_kernels(input_length, num_kernels):
    candidate_lengths = np.array((7, 9, 11))

    # initialise kernel parameters
    weights = np.zeros((num_kernels, candidate_lengths.max()))  # see note
    lengths = np.zeros(num_kernels, dtype=np.int32)  # see note
    biases = np.zeros(num_kernels)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    # note: only the first *lengths[i]* values of *weights[i]* are used

    for i in range(num_kernels):
        length = np.random.choice(candidate_lengths)
        _weights = np.random.normal(0, 1, length)
        bias = np.random.uniform(-1, 1)
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (length - 1)))
        padding = ((length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0

        weights[i, :length] = _weights - _weights.mean()
        lengths[i], biases[i], dilations[i], paddings[i] = length, bias, dilation, padding

    return weights, lengths, biases, dilations, paddings


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding):
    # zero padding
    if padding > 0:
        _input_length = len(X)
        _X = np.zeros(_input_length + (2 * padding))
        _X[padding:(padding + _input_length)] = X
        X = _X

    input_length = len(X)

    output_length = input_length - ((length - 1) * dilation)

    _ppv = 0  # "proportion of positive values"
    _max = np.NINF

    for i in range(output_length):

        _sum = bias

        for j in range(length):
            _sum += weights[j] * X[i + (j * dilation)]

        if _sum > 0:
            _ppv += 1

        if _sum > _max:
            _max = _sum

    return _ppv / output_length, _max


@njit(parallel=True, fastmath=True)
def apply_kernels(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels

    num_examples = len(X)
    num_kernels = len(weights)

    # initialise output
    _X = np.zeros((num_examples, num_kernels * 2))  # 2 features per kernel

    for i in prange(num_examples):

        for j in range(num_kernels):
            _X[i, (j * 2):((j * 2) + 2)] = \
                apply_kernel(X[i], weights[j][:lengths[j]], lengths[j], biases[j], dilations[j], paddings[j])

    return _X


class ROCKET():
    def __init__(self, num_kernels=100):

        self.num_kernels = num_kernels

    def train(self, X_train, Y_train):

        _ = generate_kernels(100, 10)
        apply_kernels(np.zeros_like(X_train)[:, 1:], _)

        input_length = X_train.shape[1]

        self.kernels = generate_kernels(input_length, self.num_kernels)
        X_transform = apply_kernels(X_train, self.kernels)
        self.classifier = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10), normalize=True)

        return self.classifier.fit(X_transform, Y_train)

    def test(self, X_test, Y_test):

        X_transform = apply_kernels(X_test, self.kernels)
        results = self.classifier.score(X_transform, Y_test)

        return results



