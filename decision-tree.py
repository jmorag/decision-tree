import numpy as np
from sklearn.model_selection import train_test_split

def information_gain(feature, labels):
    """Given a column of data and associated labels, calculate the information
    gain of splitting it on its median"""
    med = np.median(feature)
    v = np.vstack((feature, labels)).transpose()
    n = v.shape[0]
    bigger = v[v[:, 0] >= med]
    smaller = v[v[:, 0] < med]

    def entropy(vec):
        """Given matrix of [fval, label] pairs, calculate entropy"""
        n = vec.shape[0]
        if n == 0:
            return 0
        total = 0
        for c in [0.0, 1.0]:
            p = vec[vec[:, 1] == c].shape[0] / n
            total += p * p
        return 1 - total

    return entropy(v) - (
        (entropy(bigger) * bigger.shape[0] / n)
        + (entropy(smaller) * smaller.shape[0] / n)
    )


class DecTree:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def show(self, level=0, indentation=4):
        offset = " " * level * indentation
        print(offset, "Feature: ", self.feature, " Threshold: ", self.threshold)
        print(offset + "| ")
        if self.left:
            self.left.show(level + 1)
        if self.right:
            self.right.show(level + 1)


def make_tree(data, labels, epsilon=1e-10):
    if data.shape[0] <= 1 or data.shape[1] <= 1:
        return None
    if np.all(labels) or np.all(np.logical_not(labels)):
        return None
    igs = np.array([information_gain(f, labels) for f in data.T])
    if np.all(igs < epsilon):
        return None
    feature = np.argmax(igs)
    col = data[:, feature]
    median = np.median(col)
    # Partition into subsets with `feature` split on the median
    right_ixs = col >= median
    left_ixs = col < median
    # Remove the feature that we split on
    right = np.delete(data[right_ixs], feature, axis=1)
    left = np.delete(data[left_ixs], feature, axis=1)
    right_labels = labels[right_ixs]
    left_labels = labels[left_ixs]
    return DecTree(
        feature, median, make_tree(left, left_labels), make_tree(right, right_labels)
    )


def classify(row, tree):
    f = tree.feature
    t = tree.threshold
    if row[f] >= t and tree.right:
        return classify(row, tree.right)
    if row[f] >= t:
        return True
    if row[f] < t and tree.left:
        return classify(row, tree.left)
    if row[f] < t:
        return False

def classify_all(data, tree):
    return np.array([classify(row, tree) for row in data])

labels = np.loadtxt("wdbc.data", delimiter=",", usecols=0, dtype=str) == "M"
data = np.loadtxt("wdbc.data", delimiter=",", usecols=range(1, 31))

def run():
    train_labels, test_labels, train_data, test_data = train_test_split(labels, data)
    t = make_tree(train_data, train_labels)
    gen_labels = classify_all(test_data, t)
    score = np.sum(gen_labels == test_labels) / test_labels.size
    return score

def main():
    print(np.mean(np.array([run() for _ in range(100)])))

if __name__ == "__main__":
    main()
