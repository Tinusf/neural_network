import csv
import numpy as np


def read_file(file_name):
    # Reads the file and returns two numpy arrays, the first containing the feature-dataset and
    # the second list containing the labels.
    X_data = []
    y_data = []
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            label = int(float(row[-1]))
            X = np.array([float(feature) for feature in row[:-1]])
            X_data.append(X)
            y_data.append(label)
    return np.array(X_data), np.array(y_data)


def one_hot(labels, classes=10):
    out = []
    for label in labels:
        new_label = [0.0] * classes
        new_label[label] = 1.0
        out.append(new_label)
    return np.array(out)


def get_num_of_classes(y_data):
    unique = set()
    for y in y_data:
        unique.add(y)
    return len(unique)
