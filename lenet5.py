import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from typing import Any

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
        grads = {}
        loss = self.cross_entropy_loss(fw_out, labels)
        batch_size = fw_out.shape[0]
        one_hot_target = np.zeros_like(fw_out)
        one_hot_target[np.arange(batch_size), labels] = 1.0
        dZ = (fw_out - one_hot_target) / batch_size
        dW = cache["output"]["input"].T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.out_weights.T

        dZ_f6 = dX * (cache["f6"]["pre_activation"] > 0)
        dW_f6 = cache["f6"]["input"].T @ dZ_f6
        db_f6 = np.sum(dZ_f6, axis=0, keepdims=True)
        dX_f6 = dZ_f6 @ self.f84_weights.T

        dZ_c5 = dX_f6 * (cache["c5"]["pre_activation"] > 0)
        dW_c5 = cache["c5"]["input"].T @ dZ_c5
        db_c5 = np.sum(dZ_c5, axis = 0, keepdims=True)
        dX_c5 = dZ_c5 @ self.f120_weights.T

        dPool2 = np.reshape(dX_c5, (batch_size, 16, 4, 4))
        dFeatureMap2 = np.repeat(np.repeat(dPool2, 2, axis=2), 2, axis=3) / 4
        dZConv2 = dFeatureMap2 * (cache["conv2"]["pre_activation"] > 0)
        db2 = np.sum(dZConv2, axis=(0, 2, 3), keepdims=True)

        X_conv2 = cache["conv2"]["input"]
        X_windows = sliding_window_view(X_conv2, window_shape=(5, 5), axis=(2, 3))

        dW2 = np.tensordot(dZConv2, X_windows, axes=([0, 2, 3], [0, 2, 3]))
        dZ_pad = np.pad(dZConv2, pad_width=((0,0), (0,0), (4,4), (4,4)), mode='constant')
        W2_rot = self.kernel2[:, :, ::-1, ::-1]
        dZ_windows = sliding_window_view(dZ_pad, window_shape=(5, 5), axis=(2, 3))
        dX_conv2_temp = np.tensordot(dZ_windows, W2_rot, axes=([1, 4, 5], [0, 2, 3]))
        dX_conv2 = np.moveaxis(dX_conv2_temp, -1, 1)

        dPool1 = np.reshape(dX_conv2, (batch_size, 6, 12, 12))
        dFeatureMap1 = np.repeat(np.repeat(dPool1, 2, axis=2), 2, axis=3) / 4
        dZConv1 = dFeatureMap1 * (cache["conv1"]["pre_activation"] > 0)
        db1 = np.sum(dZConv1, axis=(0, 2, 3), keepdims=True)

        X_conv1 = cache["conv1"]["input"]
        X_windows = sliding_window_view(X_conv1, window_shape=(5, 5), axis=(2, 3))

        dW1 = np.tensordot(dZConv1, X_windows, axes=([0, 2, 3], [0, 2, 3]))
        dZ_pad = np.pad(dZConv1, pad_width=((0,0), (0,0), (4,4), (4,4)), mode='constant')
        W1_rot = self.kernel1[:, :, ::-1, ::-1]
        dZ_windows = sliding_window_view(dZ_pad, window_shape=(5, 5), axis=(2, 3))
        dX_conv1_temp = np.tensordot(dZ_windows, W1_rot, axes=([1, 4, 5], [0, 2, 3]))
        dX_conv1 = np.moveaxis(dX_conv1_temp, -1, 1)

        grads["kernel1"]      = dW1
        grads["bias1"]        = db1

        grads["kernel2"]      = dW2
        grads["bias2"]        = db2

        grads["f120_weights"] = dW_c5
        grads["f120_bias"]    = db_c5

        grads["f84_weights"]  = dW_f6
        grads["f84_bias"]     = db_f6

        grads["out_weights"]  = dW
        grads["out_bias"]     = db

        return grads, loss

    def update_weights(self, grads: dict[str, NDArray[np.float64]], learning_rate: float):
            self.kernel1 -= learning_rate * grads["kernel1"]
            self.bias1   -= learning_rate * grads["bias1"]

            self.kernel2 -= learning_rate * grads["kernel2"]
            self.bias2   -= learning_rate * grads["bias2"]

            self.f120_weights -= learning_rate * grads["f120_weights"]
            self.f120_bias    -= learning_rate * grads["f120_bias"]

            self.f84_weights  -= learning_rate * grads["f84_weights"]
            self.f84_bias     -= learning_rate * grads["f84_bias"]

            self.out_weights  -= learning_rate * grads["out_weights"]
            self.out_bias     -= learning_rate * grads["out_bias"]

    def cross_entropy_loss(self, probabilities: NDArray[np.float64], labels: NDArray[np.uint8]) -> np.float64:
        one_hot_target = np.zeros_like(probabilities)
        batch_size = probabilities.shape[0]
        one_hot_target[np.arange(batch_size), labels] = 1.0

        epsilon = 1e-15
        clipped_probs = np.clip(probabilities, epsilon, 1.0 - epsilon)

        loss = -np.sum(one_hot_target * np.log(clipped_probs)) / batch_size
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
