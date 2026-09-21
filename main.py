import numpy as np
import load, lenet5

def main():
    x_train = load.images("train-images.idx3-ubyte")
    y_train = load.labels("train-labels.idx1-ubyte")

    x_test = load.images("t10k-images.idx3-ubyte")
    y_test = load.labels("t10k-labels.idx1-ubyte")

    slice = 5000
    batch_size = 32
    num_tests = 1000

    x_train = np.float64(x_train[:slice]/255)
    data = np.reshape(x_train, (slice, 1, x_train.shape[1], x_train.shape[2]))
    labels = y_train[:slice]

    x_test = np.float64(x_test[:num_tests]/255)
    test_data = np.reshape(x_test, (num_tests, 1, x_test.shape[1], x_test.shape[2]))
    test_labels = y_test[:num_tests]

    net = lenet5.LeNet5()

    learning_rate = 0.01
    epochs = 40

    num_samples = data.shape[0]

    for epoch in range(epochs):
        indicies = np.arange(num_samples)
        np.random.shuffle(indicies)
        data_shuffled = data[indicies]
        label_shuffled = labels[indicies]

        epoch_loss = 0.0
        correct_predictions = 0

        for i in range(0, num_samples, batch_size):
            x_batch = data_shuffled[i:i + batch_size]
            y_batch = label_shuffled[i:i + batch_size]

            out, cache = net.forward(x_batch)
            grads, batch_loss = net.backward(out, cache, y_batch)
            net.update_weights(grads, learning_rate)

            epoch_loss += batch_loss * x_batch.shape[0]

            predictions = np.argmax(out, axis=1)
            correct_predictions += np.sum(predictions == y_batch)

        avg_loss = epoch_loss / num_samples
        accuracy = (correct_predictions / num_samples) * 100
        print(f"Epoch {epoch:02d} | AVG Loss: {avg_loss:.4f} | Accuracy: %{accuracy:.0f}")

    out, cache = net.forward(test_data)
    predictions = np.argmax(out, axis=1)
    correct_predictions = np.sum(predictions == test_labels)
    accuracy = (correct_predictions/num_tests)*100
    print("Test Accuracy for ", num_tests, " images: ", accuracy)



if __name__ == "__main__":
    main()
