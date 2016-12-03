# CART on the Bank Note dataset
from random import seed
from random import randrange,randint
from csv import reader
from numpy import array


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
def cross_validation_split(data_cross, n_folds):
    dataset_split = list()
    dataset_copy = list(data_cross)
    print (dataset_copy)
    fold_size = len(data_cross) / n_folds
    for i in range(n_folds):
        fold = list()
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
def evaluate_algorithm(data, algorithm, n_folds, *args):
    print("EVALUATE ALGORITHM", data)
    # folds = cross_validation_split(data, n_folds)
    # scores = list()
    # trees = list()
    # for fold in folds:
    #     train_set = list(folds)
    #     train_set.remove(fold)
        # print ("PREVIOUS TRAIN SET", train_set)
        # train_set = sum(train_set, [])
        # print ("TRAIN SET", train_set)
        # test_set = list()
        # for row in fold:
        #     row_copy = list(row)
        #     test_set.append(row_copy)
        #     row_copy[-1] = None
    print("LETS GO TO DECISION TREE")
    tree = algorithm(data,  *args)
        # trees.append(tree)
        # actual = [row[-1] for row in fold]
        # accuracy = accuracy_metric(actual, predicted)
        # scores.append(accuracy)
    return tree


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
def get_split(gdata):
    class_values = list(set(row[-1] for row in gdata))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(gdata[0]) - 1):
        for row in gdata:
            groups = test_split(index, row[index], gdata)
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
def build_tree(ddata,  max_depth, min_size):
    root = get_split(ddata)
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
def decision_tree(ddata, max_depth, min_size):
    print("DECISION TREE")
    tree = build_tree(ddata, max_depth, min_size)
    # predictions = list()
    # for row in test:
    #     prediction = predict(tree, row)
    #     predictions.append(prediction)
    #     print ("TEST ROW", row)
    #     print ("PREDICTION", prediction)
    return (tree)


def polling(a):
    p = list()

    print (len(a))

    for i in range(len(a)):
        zero =0.0
        one =0.0
        two =0.0
        three =0.0
        for j in range(len(a[i])):
            if(a[i][j] == 0.0):
                zero+=1.0
            elif(a[i][j] == 1.0):
                one+=1.0
            elif(a[i][j] == 2.0):
                two+=1.0
            elif(a[i][j] == 3.0):
                three+=1.0
        if(max(zero,one,two,three)==zero):
            p.append(0.0)
        elif(max(zero,one,two,three) == one):
            p.append(1.0)
        elif (max(zero, one, two, three) == two):
            p.append(2.0)
        elif (max(zero, one, two, three) == three):
            p.append(3.0)

    return p


def testPrediction(test_d_pred, Decision_T):
    result = list()
    for t in test_d_pred:
        p = list()
        for r in Decision_T:
            pred = predict(r, t)
            p.append(pred)
        result.append(p)

    final_pred = array(result)

    return final_pred

filename = 'data2.csv'
datafile = load_csv(filename)


new_dataset = list()

n_folds = 5
max_depth = 5
min_size = 10
row = 0
m = 10
for i in range(20):
    d = list()
    while(row < (500 * (i +1))):
        d.append(datafile[row])
        row += 1
    new_dataset.append(d)



dTree = list()
print("TRAINING PART     ")
for i in range(len(new_dataset)):
# for i in range(1):
    print (new_dataset[i])

    for j in range(len(new_dataset[i][0])):
        str_column_to_float(new_dataset[i], j)

    # evaluate algorithm
    tree = evaluate_algorithm(new_dataset[i], decision_tree, n_folds, max_depth, min_size)
    dTree.append(tree)


print("TESTING PART")

testfilename = 'testDT.csv'
tData = load_csv(testfilename)

for j in range(len(tData[0])):
    str_column_to_float(tData, j)

tLabel = list()
for j in range(m):
    tLabel.append(tData[j][-1])
    tData[j][-1]= None

fpred = testPrediction(tData, dTree)
new_pred = polling(fpred)

posc = 0
for i in range(len(tLabel)):
    if(tLabel[i] == new_pred[i]):
        posc+=1


print ("ACCURACY : ", posc/len(tLabel) * 100)