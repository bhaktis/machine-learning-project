# CART on the Bank Note dataset
from random import seed
from random import randrange,randint
from csv import reader


# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    print (dataset)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    print (dataset_copy)
    fold_size = len(dataset) / n_folds
    print ("FOLD: ",n_folds)
    print ("FOLD_SIZE: ",fold_size)
    for i in range(n_folds):
        fold = list()
        print ("FOLD INSIDE", fold)
        while len(fold) < fold_size and len(dataset_copy) > 0:
            index = randint(0, (len(dataset_copy)-1))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    print("EVALUATE ALGORITHM")
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    trees = list()
    for fold in folds:
        print ("FOLD VALUE" ,fold)
        train_set = list(folds)
        train_set.remove(fold)
        print ("PREVIOUS TRAIN SET", train_set)
        train_set = sum(train_set, [])
        print ("TRAIN SET", train_set)
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        print("LETS GO TO DECISION TREE")
        print("TRAIN SET", train_set)
        tree = algorithm(train_set, *args)
        trees.append(tree)
        # actual = [row[-1] for row in fold]
        # accuracy = accuracy_metric(actual, predicted)
        # scores.append(accuracy)
    return trees


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    print("CLASS VALUES" ,class_values)
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    print (dataset)
    print(dataset[0])
    print(len(dataset[0]))
    print(range(len(dataset[0]) - 1))
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    print ("SPLIT")
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(dataset)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Classification and Regression Tree Algorithm
def decision_tree(train, max_depth, min_size):
    print("DECISION TREE")
    tree = build_tree(train, max_depth, min_size)
    # predictions = list()
    # for row in test:
    #     prediction = predict(tree, row)
    #     predictions.append(prediction)
    #     print ("TEST ROW", row)
    #     print ("PREDICTION", prediction)
    return tree


# Test CART on Bank Note dataset
seed(1)
# load and prepare data
# filename = 'data_banknote_authentication.csv'
filename = 'data2.csv'
dataset = load_csv(filename)
# convert string attributes to integers

new_dataset = []

n_folds = 5
max_depth = 5
min_size = 10
row = 0;
for i in range(20):
    d = list()
    while(row < (500 * (i +1))):
        d.append(dataset[row])
        row += 1
    new_dataset.append(d)


for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)


decision_tree = list()

for i in range(len(new_dataset)):
    print (new_dataset[i])

    # for j in range(len(new_dataset[i][0])):
    #     str_column_to_float(new_dataset[i], j)

    # evaluate algorithm

    print("TRAINING PHASE")

    print("OLD DATASET", dataset)

    tree = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
    decision_tree.append(tree)
    #
    # print('Tree', trees)
    #
    # print("TESTING PHASE")
    # test_data = [[78,50,66,75,77,75,75,68,75,77,5,7,19,15,10,None], [55,67,69,58,59,67,47,25,20,21,7,12,31,10,15, None]]
    #
    # predictions = list()
    # for t in test_data:
    #     for r in trees:
    #         pred = predict(r,t)
    #         predictions.append(pred)
    #
    # print("PREDICTIONS", predictions)
    #
    #
    # print('Scores: %s' % scores)
    # print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


print (decision_tree)