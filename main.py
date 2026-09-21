import numpy as np
import load, lenet5

def main():
    x_train = load.images("train-images.idx3-ubyte")
    y_train = load.labels("train-labels.idx1-ubyte")

    x_test = load.images("t10k-images.idx3-ubyte")
    y_test = load.labels("t10k-labels.idx1-ubyte")

if __name__ == "__main__":
    main()
