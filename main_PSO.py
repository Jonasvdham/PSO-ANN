import csv
import time
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import PSO
import functools
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection


def load_data(no_samples):
    global train_imgs, test_imgs, train_labels, test_labels, train_labels_one_hot, test_labels_one_hot, X, X_test, y, y_test
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
    X = train_imgs
    X_test = test_imgs
    y = train_labels.flatten().astype(int)
    y_test = test_labels.flatten().astype(int)
    print("Data loaded.", np.shape(X))
    print("Train", len(train_imgs), "Test", len(test_imgs))


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


class MultiLayerPerceptron:
    def __init__(self, shape, weights):
        self.shape = shape
        self.num_layers = len(shape)
        self.weights = weights

    def run(self, data):
        layer = data.T
        for i in range(self.num_layers - 1):
            prev_layer = np.insert(
                layer, np.shape(layer)[0], 1, axis=0
            )  # matrix method
            o = np.dot(self.weights[i], prev_layer)
            layer = 1 / (1 + np.exp(-o))
        return layer


def dim_weights(shape):
    dim = 0
    for i in range(len(shape) - 1):
        dim = dim + (shape[i] + 1) * shape[i + 1]
    return dim


def weights_to_vector(weights):
    w = np.asarray([])
    for i in range(len(weights)):
        v = weights[i].flatten()
        w = np.append(w, v)
    return w


def vector_to_weights(vector, shape):
    weights = []
    idx = 0
    for i in range(len(shape) - 1):
        r = shape[i + 1]
        c = shape[i] + 1
        idx_min = idx
        idx_max = idx + r * c
        W = vector[idx_min:idx_max].reshape(r, c)
        weights.append(W)
        idx = idx_max
    return weights


def eval_neural_network(weights, shape, X, y):
    mse = np.asarray([])
    for particle in weights:
        weights = vector_to_weights(particle, shape)
        nn = MultiLayerPerceptron(shape, weights=weights)
        y_pred = nn.run(X)
        mse = np.append(
            mse, sklearn.metrics.mean_squared_error(y, y_pred.T)
        )  # MSE
    return mse


def eval_accuracy(weights, shape, X, y):
    corrects, wrongs = 0, 0
    nn = MultiLayerPerceptron(shape, weights=weights)
    predictions = []
    for i in range(len(X)):
        out_vector = nn.run(X[i])
        y_pred = np.argmax(out_vector)
        predictions.append(y_pred)
        if y_pred == y[i]:
            corrects += 1
        else:
            wrongs += 1
    return corrects, wrongs, predictions


def run_test(test_number, hidden_layers, Export=False):
    # Start timing
    time_started = time.strftime("%Y-%m-%d-%H-%M-%S")

    num_inputs = X.shape[1]

    # Variables
    no_iterations = 4000
    shape = [num_inputs] + hidden_layers + [10]
    no_particles = 25
    swarm_inertia = 0.9
    swarm_phi_p = 1
    swarm_phi_g = 3
    swarm_v_max = 3
    accuracies = [0.1]
    measured_times = [
        time_delta(time_started, time.strftime("%Y-%m-%d-%H-%M-%S"))
    ]

    # Initialize swarm
    cost_func = functools.partial(
        eval_neural_network, shape=shape, X=X, y=train_labels_one_hot
    )
    swarm = PSO.ParticleSwarm(
        cost_func,
        num_dimensions=dim_weights(shape),
        num_particles=no_particles,
        inertia=swarm_inertia,
        phi_p=swarm_phi_p,
        phi_g=swarm_phi_g,
        v_max=swarm_v_max,
    )

    # Train...
    i = 0
    best_scores = [(i, swarm.best_score)]
    while swarm.best_score > 1e-6 and i < no_iterations:
        swarm.update(num_iterations=no_iterations, current_iter=i + 1)
        i = i + 1
        # if swarm.best_score < best_scores[-1][1]:
        if i % 10 == 0:
            best_scores.append((i, swarm.best_score))
            corrects, wrongs, predictions = eval_accuracy(
                vector_to_weights(swarm.g, shape), shape, X_test, y_test
            )
            accuracy = corrects / (corrects + wrongs)
            current_time = time_delta(
                time_started, time.strftime("%Y-%m-%d-%H-%M-%S")
            )
            accuracies.append(accuracy)
            measured_times.append(current_time)
            print(
                best_scores[-1][0],
                "- Time(s):",
                current_time,
                "Error:",
                best_scores[-1][1],
                "Accuracy:",
                accuracy,
            )

    # End time
    time_ended = time.strftime("%Y-%m-%d-%H-%M-%S")
    time_elapsed = time_delta(time_started, time_ended)

    # Plot
    iters = [tup[0] for tup in best_scores]
    errors = [tup[1] for tup in best_scores]
    """
    figure = plt.figure()
    errorplot = plt.subplot(2, 1, 1)
    errorplot.plot(iters, errors)
    plt.title("PSO")
    plt.ylabel("Mean squared error")

    accuracyplot = plt.subplot(2, 1, 2)
    accuracyplot.plot(iters, accuracies)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    """

    # Test...
    best_weights = vector_to_weights(swarm.g, shape)
    best_nn = MultiLayerPerceptron(shape, weights=best_weights)
    y_test_pred = np.round(best_nn.run(X_test))
    print(
        sklearn.metrics.classification_report(
            test_labels_one_hot, y_test_pred.T
        )
    )
    print("Accuracy:", accuracy)
    print("Time(s):", time_elapsed)

    # Output results
    report = sklearn.metrics.classification_report(
        test_labels_one_hot, y_test_pred.T, output_dict=True
    )
    report = DataFrame(report).transpose().values.tolist()

    # start CSV file
    ExportToFile = (
        "results/PSO/"
        + str(len(train_imgs))
        + str(hidden_layers)
        + "/"
        + str(test_number)
        + "-accuracy-"
        + str(accuracy)
        + "-time-"
        + time_started
        + ".csv"
    )
    if Export == True:

        # figure.savefig("results/PSO/figures/"+str(len(train_imgs))+str(hidden_layers)+"/"+str(test_number)+"-"+"-time-"+time_started+".png")
        # plt.clf()
        with open(ExportToFile, "a", newline="\n") as out:
            writer = csv.writer(out, delimiter=",")

            writer.writerow(["Accuracies"] + accuracies)
            writer.writerow(["Errors"] + errors)
            writer.writerow(["Iters"] + iters)
            writer.writerow(["Time(s)"] + measured_times)
            writer.writerow([])
            header = [
                "#Particles",
                "#Iterations",
                "Inertia",
                "Phi_p",
                "Phi_g",
                "V_max",
                "Accuracy",
                "Time Elapsed (s)",
                "Inputs",
                "Hidden nodes",
                "Outputs",
                "Time started",
                "Time ended",
            ]
            writer.writerow(header)
            line = [
                no_particles,
                no_iterations,
                swarm_inertia,
                swarm_phi_p,
                swarm_phi_g,
                swarm_v_max,
                accuracy,
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
for i in range(6):
    run_test(i + 14, [15], Export=False)
    run_test(i + 14, [64, 32], Export=False)
