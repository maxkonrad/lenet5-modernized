import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from typing import Any
import load

class LeNet5:
    def __init__(self,) -> None:
        self.kernel1 : NDArray[np.float64] = np.random.normal(loc=0.0, scale=0.1, size=(6, 1, 5, 5))
        self.kernel2 : NDArray[np.float64] = np.random.normal(loc=0.0, scale=0.1, size=(16, 6, 5, 5))

        self.bias1 : NDArray[np.float64] = np.zeros(shape=(1, 6, 1, 1))
        self.bias2 : NDArray[np.float64] = np.zeros(shape=(1, 16, 1, 1))

        self.f120_weights : NDArray[np.float64] = np.random.normal(loc=0.0, scale=0.1, size=(256, 120))
        self.f120_bias :NDArray[np.float64] = np.random.normal(loc=0.0, scale=0.1, size=(1, 120))

        self.f84_weights : NDArray[np.float64] = np.random.normal(loc=0.0, scale=0.1, size=(120, 84))
        self.f84_bias : NDArray[np.float64] = np.random.normal(loc=0.0, scale=0.1, size=(1, 84))

        self.out_weights : NDArray[np.float64] = np.random.normal(loc=0.0, scale=0.1, size=(84, 10))
        self.out_bias : NDArray[np.float64]= np.random.normal(loc=0.0, scale=0.1, size=(1, 10))

    def forward(self, data: NDArray[np.float64]) -> tuple[NDArray[np.float64], dict[str, Any]]:
        cache = {}
        cache["conv1"] = {}
        cache["pool1"] = {}
        cache["conv2"] = {}
        cache["pool2"] = {}
        cache["c5"] = {}
        cache["f6"] = {}
        cache["output"] = {}

        cache["conv1"]["input"] = data
        feature_map1, pre_activation1 = self.conv2D(data, self.kernel1, self.bias1)
        cache["conv1"]["pre_activation"] = pre_activation1
        #print(feature_map1.shape)
        cache["pool1"]["input"] = feature_map1
        pool1 = self.pool(feature_map1)
        #print(pool1.shape)
        feature_map2, pre_activation2 = self.conv2D(pool1, self.kernel2, self.bias2)
        cache["conv2"]["input"] = pool1
        cache["conv2"]["pre_activation"] = pre_activation2
        #print(feature_map2.shape)
        cache["pool2"]["input"] = feature_map2
        pool2 = self.pool(feature_map2)
        #print(pool2.shape)
        flattened = self.flatten(pool2)
        #print(flattened.shape)
        c5_pre = self.fully_connect(flattened, self.f120_weights, self.f120_bias)
        c5_out = self.activate(c5_pre)
        cache["c5"]["input"] = flattened
        cache["c5"]["pre_activation"] = c5_pre
        #print(c5_out.shape)
        f6_pre = self.fully_connect(c5_out, self.f84_weights, self.f84_bias)
        f6_out = self.activate(f6_pre)
        cache["f6"]["input"] = c5_out
        cache["f6"]["pre_activation"] = f6_pre
        #print(f6_out.shape)
        logits = self.fully_connect(f6_out, self.out_weights, self.out_bias)
        cache["output"]["input"] = f6_out
        cache["output"]["logits"] = logits
        out = self.softmax(logits)
        #print(out.shape)
        return out, cache

    def backward(self, fw_out: NDArray[np.float64], cache: dict[str, Any], labels: NDArray[np.uint8]):
        self.cross_entropy_loss(fw_out, labels)

    def cross_entropy_loss(self, probabilities: NDArray[np.float64], labels: NDArray[np.uint8]) -> NDArray[np.float64]:
        one_hot_target = np.zeros_like(probabilities)
        batch_size = probabilities.shape[0]
        one_hot_target[np.arange(batch_size), labels] = 1.0

        epsilon = 1e-15
        clipped_probs = np.clip(probabilities, epsilon, 1.0 - epsilon)

        loss : NDArray[np.float64] = -np.sum(one_hot_target * np.log(clipped_probs)) / batch_size
        return loss

    def conv2D(self, input: NDArray[np.float64], kernel: NDArray[np.float64], bias: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        windows = sliding_window_view(input, window_shape=kernel.shape[2:], axis=(2,3))
        feature_map = np.tensordot(windows, kernel, axes=([1, 4, 5], [1, 2, 3]),)
        torchlike_fm = np.moveaxis(feature_map, -1, 1)
        pre_activation = torchlike_fm + bias
        activated = self.activate(pre_activation)
        return activated, pre_activation

    def activate(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        return (np.maximum(0, input))

    def pool(self, input: NDArray[np.float64]):
        input = input.reshape((input.shape[0], input.shape[1], input.shape[2]//2, 2, input.shape[3]//2, 2))
        mean = np.mean(input, axis=(3, 5))
        return mean

    def flatten(self, input: NDArray[np.float64]):
        output = input.reshape(input.shape[0], -1)
        return output

    def fully_connect(self, flattened: NDArray[np.float64], weights: NDArray[np.float64], bias: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.dot(flattened, weights) + bias
        return out

    def softmax(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted_logits = input - np.max(input, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probabilities


x_train = load.images("train-images.idx3-ubyte")
y_train = load.labels("train-labels.idx1-ubyte")


data: NDArray[np.float64] = np.random.rand(1, 1, 28, 28)
net = LeNet5()
out, cache = net.forward(data)
net.backward(out, cache, y_train)
