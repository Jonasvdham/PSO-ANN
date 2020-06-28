import numpy as np

training_set_size = 4000


# Encode MNIST data into numpy binary for faster access
train_data = np.loadtxt("mnist_train.csv", delimiter=",")
test_data = np.loadtxt("mnist_test.csv", delimiter=",")
train_imgs = np.asfarray(train_data[:training_set_size, 1:]) / 255.0
test_imgs = (
    np.asfarray(test_data[: int(np.floor(training_set_size / 4)), 1:]) / 255.0
)
train_labels = np.asfarray(train_data[:training_set_size, :1])
test_labels = np.asfarray(
    test_data[: int(np.floor(training_set_size / 4)), :1]
)
lr = np.arange(10)
train_labels_one_hot = (lr == train_labels).astype(np.float)
test_labels_one_hot = (lr == test_labels).astype(np.float)

for i, d in enumerate(
    [
        train_imgs,
        test_imgs,
        train_labels,
        test_labels,
        train_labels_one_hot,
        test_labels_one_hot,
    ]
):
    np.save("data/%s/%i.array" % (str(training_set_size), i), d)
