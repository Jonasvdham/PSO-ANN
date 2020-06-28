from __future__ import division
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import (
    expit as activation_function,
)  # 1/(1+exp(-x)), sigmoid
from scipy.stats import truncnorm
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, classification_report


def load_data(no_samples):
    global train_imgs, test_imgs, train_labels, test_labels, train_labels_one_hot, test_labels_one_hot, no_training_samples
    (
        train_imgs,
        test_imgs,
        train_labels,
        test_labels,
        train_labels_one_hot,
        test_labels_one_hot,
    ) = [
        np.load("data/%d/%i.array.npy" % (int(no_samples), i))
        for i in range(6)
    ]
    print("Data loaded.")
    no_training_samples = len(train_imgs)
    print("Train", no_training_samples, "Test", len(test_imgs))


def time_delta(time_started, current_time):
    start_sec = (
        int(time_started[8:10]) * 24 * 3600
        + int(time_started[11:13]) * 3600
        + int(time_started[14:16]) * 60
        + int(time_started[17:19])
    )
    end_sec = (
        int(current_time[8:10]) * 24 * 3600
        + int(current_time[11:13]) * 3600
        + int(current_time[14:16]) * 60
        + int(current_time[17:19])
    )
    return end_sec - start_sec


def one_hot(array):
    output = np.zeros((len(array), 10))
    output[np.arange(len(array)), array.astype(int).tolist()] = 1
    return output


class NeuralNetwork:
    def __init__(self, network_structure, learning_rate, bias=None):
        self.structure = network_structure
        self.no_of_layers = len(self.structure)
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    # Initialization
    def create_weight_matrices(self):
        bias_node = 1 if self.bias else 0
        self.weights_matrices = []
        for k in range(self.no_of_layers - 1):
            nodes_in = self.structure[k]
            nodes_out = self.structure[k + 1]
            n = (nodes_in + bias_node) * nodes_out
            X = truncnorm(-1, 1, loc=0, scale=1)
            wm = X.rvs(n).reshape((nodes_out, nodes_in + bias_node))
            self.weights_matrices.append(wm)

    def train(self, input_vector, target_vector):
        input_vector = np.array(input_vector, ndmin=2).T
        res_vectors = [input_vector]
        for k in range(self.no_of_layers - 1):
            in_vector = res_vectors[-1]
            if self.bias:
                in_vector = np.concatenate((in_vector, [[self.bias]]))
                res_vectors[-1] = in_vector
            x = np.dot(self.weights_matrices[k], in_vector)
            out_vector = activation_function(x)
            res_vectors.append(out_vector)

        target_vector = np.array(target_vector, ndmin=2).T
        output_errors = target_vector - out_vector
        for k in range(self.no_of_layers - 1, 0, -1):
            out_vector = res_vectors[k]
            in_vector = res_vectors[k - 1]
            if self.bias and not k == (self.no_of_layers - 1):
                out_vector = out_vector[:-1, :].copy()
            tmp = (
                output_errors * out_vector * (1.0 - out_vector)
            )  # sigma'(x) = sigma(x) (1 - sigma(x))
            tmp = np.dot(tmp, in_vector.T)
            self.weights_matrices[k - 1] += self.learning_rate * tmp
            output_errors = np.dot(
                self.weights_matrices[k - 1].T, output_errors
            )
            if self.bias:
                output_errors = output_errors[:-1, :]

    def run(self, input_vector):
        if self.bias:
            input_vector = np.concatenate((input_vector, [self.bias]))
        in_vector = np.array(input_vector, ndmin=2).T
        for k in range(self.no_of_layers - 1):
            x = np.dot(self.weights_matrices[k], in_vector)
            out_vector = activation_function(x)
            in_vector = out_vector
            if self.bias:
                in_vector = np.concatenate((in_vector, [[self.bias]]))
        return out_vector

    def eval_accuracy(self, data, labels):
        results = self.run_set(data)
        n = len(labels)
        corrects = (results == (np.reshape(labels, n))).sum()
        wrongs = n - corrects
        accuracy = corrects / (corrects + wrongs)
        return corrects, wrongs, accuracy

    def eval_error(self, data, labels):
        results = np.array(self.run(data[0]).T)
        for i in range(len(data) - 1):
            results = np.append(results, (self.run(data[i + 1])).T, axis=0)
        error = mean_squared_error(labels, results)
        return error

    def run_set(self, data):
        results = np.array([]).reshape((10, 0))
        for i in range(len(data)):
            results = np.append(results, self.run(data[i]).argmax())
        return results


def run_test(test_number, no_epochs, hidden_layers, Export=False):
    # Start timing
    time_started = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Parameter settings
    num_inputs = len(train_imgs[0])
    shape = [num_inputs] + hidden_layers + [10]
    ANN = NeuralNetwork(network_structure=shape, learning_rate=0.01, bias=1)

    # Main loop
    accuracies = []
    measured_times = []
    errors = []
    for epoch in range(no_epochs):
        for i in range(no_training_samples):
            if i % 1000 == 0:
                print(
                    "epoch:", epoch, "img number:", i, "/", no_training_samples
                )
                corrects, wrongs, accuracy = ANN.eval_accuracy(
                    test_imgs, test_labels
                )
                error = ANN.eval_error(test_imgs, test_labels_one_hot)
                current_time = time_delta(
                    time_started, time.strftime("%Y-%m-%d-%H-%M-%S")
                )
                accuracies.append(accuracy)
                errors.append(error)
                measured_times.append(current_time)
            ANN.train(train_imgs[i], train_labels_one_hot[i])

    # End time
    time_ended = time.strftime("%Y-%m-%d-%H-%M-%S")
    time_elapsed = time_delta(time_started, time_ended)

    corrects, wrongs, accuracy = ANN.eval_accuracy(test_imgs, test_labels)
    print("accuracy: test", accuracy)
    error = ANN.eval_error(test_imgs, test_labels_one_hot)
    print("MSE: ", error)

    """
    figure = plt.figure()
    mseplot = plt.subplot(2, 1, 1)
    mseplot.plot(errors)
    plt.title("BP - MSE and Accuracy")
    plt.ylabel("Mean squared error")


    accplot = plt.subplot(2, 1, 2)
    accplot.plot(accuracies)
    plt.xlabel("Samples (x1000)")
    plt.ylabel("Accuracy")
    """

    y_test_pred = ANN.run_set(test_imgs)
    y_test_pred_one_hot = one_hot(y_test_pred)
    print(classification_report(test_labels_one_hot, y_test_pred_one_hot))

    # Results output
    if Export:
        # figure.savefig("results/BP/figures/"+str(no_training_samples)+"-"+str(test_number)+"-"+str(hidden_layers)+"-epochs-"+str(no_epochs)+"-time-"+time_started+".png")
        # plt.clf()
        report = classification_report(
            test_labels_one_hot, y_test_pred_one_hot, output_dict=True
        )
        report = DataFrame(report).transpose().values.tolist()

        # start CSV file
        ExportToFile = (
            "results/BP/"
            + str(no_training_samples)
            + str(hidden_layers)
            + "/"
            + str(test_number)
            + "-accuracy-"
            + str(accuracy)
            + "-time-"
            + time_started
            + ".csv"
        )
        with open(ExportToFile, "a", newline="\n") as out:
            writer = csv.writer(out, delimiter=",")

            writer.writerow(["Accuracies"] + accuracies)
            writer.writerow(["Errors"] + errors)
            writer.writerow(["Time(s)"] + measured_times)
            writer.writerow([])

            header = [
                "#Epochs",
                "MSE",
                "Accuracy",
                "Train_imgs",
                "Time Elapsed (s)",
                "Inputs",
                "Hidden nodes",
                "Layer 2",
                "Time started",
                "Time ended",
            ]
            writer.writerow(header)
            line = [
                no_epochs,
                error,
                accuracy,
                no_training_samples,
                time_elapsed,
                shape[0],
                shape[1],
                shape[2],
                time_started,
                time_ended,
            ]
            writer.writerow(line)
            writer.writerow("\n")
            writer.writerow(
                ["#", "Precision", "Recall", "f1-score", "support"]
            )
            for x in range(10):
                writer.writerow(
                    [x, report[x][0], report[x][1], report[x][2], report[x][3]]
                )
            writer.writerow(
                [
                    "Micro avg",
                    report[10][0],
                    report[10][1],
                    report[10][2],
                    report[10][3],
                ]
            )
            writer.writerow(
                [
                    "Macro avg",
                    report[11][0],
                    report[11][1],
                    report[11][2],
                    report[11][3],
                ]
            )
            writer.writerow(
                [
                    "Weighted avg",
                    report[12][0],
                    report[12][1],
                    report[12][2],
                    report[12][3],
                ]
            )
            writer.writerow(
                [
                    "Samples avg",
                    report[13][0],
                    report[13][1],
                    report[13][2],
                    report[13][3],
                ]
            )
        out.close()


load_data(4215)

run_test(0, 20, hidden_layers=[15], Export=False)
run_test(0, 20, hidden_layers=[64, 32], Export=False)
