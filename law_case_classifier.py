from __future__ import absolute_import, division, print_function

import csv
import random
import math
import operator
import codecs
import glob


def load_dataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] is predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def run():
    # prepare data
    training_set = []
    test_set = []
    split = 0.67

    cases = sorted(glob.glob("./dataset/*.txt"))

    corpus_raw = u""
    for case in cases:
        with codecs.open(case, "r", "utf-8") as case_file:
            corpus_raw += case_file.read()

    load_dataset(corpus_raw, split, training_set, test_set)
    print('Train set: ' + repr(len(training_set)))
    print('Test set: ' + repr(len(test_set)))

    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(test_set[x][-1]))
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

run()