import numpy as np
import matplotlib.pyplot as plt


def load_knn_data():
    test_inputs = np.genfromtxt('knn-dataset/test_inputs.csv', delimiter=','),
    test_labels = np.genfromtxt('knn-dataset/test_labels.csv', delimiter=','),
    train_inputs = np.genfromtxt('knn-dataset/train_inputs.csv', delimiter=','),
    train_labels = np.genfromtxt('knn-dataset/train_labels.csv', delimiter=','),
    return train_inputs, train_labels, test_inputs, test_labels


'''
This function implements the KNN classifier to predict the label of a data point. 
Measure distances with the Euclidean norm (L2 norm).  
When there is a tie between two (or more) labels, break the tie by choosing any label.

Inputs:
    **x**: input data point for which we want to predict the label (numpy array of M features)
    **inputs**: matrix of data points in which neighbors will be found (numpy array of N data points x M features)
    **labels**: vector of labels associated with the data points  (numpy array of N labels)
    **k_neighbors**: # of nearest neighbors that will be used
Outputs:
    **predicted_label**: predicted label (integer)
'''   
def predict_knn(x, inputs, labels, k_neighbors):
    predicted_label = 0
    ########
    # TO DO:
    # preprocess for wrong data
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    norm_labels = []    # [(dist, label), ...]
    for i, l in zip(inputs, labels):
        norm = np.linalg.norm(x - i)
        norm_labels.append((norm, l))
    norm_labels.sort()

    knn = norm_labels[:k_neighbors]
    cnt = {}
    for _, label in knn:
        cnt[label] = cnt.get(label, 0) + 1
    predicted_label = max(cnt.items(), key=lambda x: x[1])[0]
    ########
    return predicted_label


'''
This function evaluates the accuracy of the KNN classifier on a dataset. 
The dataset to be evaluated consists of (inputs, labels). 
The dataset used to find nearest neighbors consists of (train_inputs, train_labels).

Inputs:
   **inputs**: matrix of input data points to be evaluated (numpy array of N data points x M features)
   **labels**: vector of target labels for the inputs (numpy array of N labels)
   **train_inputs**: matrix of input data points in which neighbors will be found (numpy array of N' data points x M features)
   **train_labels**: vector of labels for the training inputs (numpy array of N' labels)
   **k_neighbors**: # of nearest neighbors to be used (integer)
Outputs:
   **accuracy**: percentage of correctly labeled data points (float)
'''
def eval_knn(inputs, labels, train_inputs, train_labels, k_neighbors):
    accuracy = 0
    ########
    # TO DO:
    # preprocess for wrong data
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    if isinstance(train_inputs, tuple):
        train_inputs = train_inputs[0]
    if isinstance(labels, tuple):
        labels = labels[0]
    if isinstance(train_labels, tuple):
        train_labels = train_labels[0]

    cnt = 0
    for x, l in zip(inputs, labels):
        predicted_label = predict_knn(x, train_inputs, train_labels, k_neighbors)
        if predicted_label == l:
            cnt += 1
    accuracy = (cnt / len(inputs)) * 100
    ########
    return accuracy


'''
This function performs k-fold cross validation to determine the best number of neighbors for KNN.
        
Inputs:
    **k_folds**: # of folds in cross-validation (integer)
    **hyperparameters**: list of hyperparameters where each hyperparameter is a different # of neighbors (list of integers)
    **inputs**: matrix of data points to be used when searching for neighbors (numpy array of N data points by M features)
    **labels**: vector of labels associated with the inputs (numpy array of N labels)
Outputs:
    **best_hyperparam**: best # of neighbors for KNN (integer)
    **best_accuracy**: accuracy achieved with best_hyperparam (float)
    **accuracies**: vector of accuracies for the corresponding hyperparameters (numpy array of floats)
'''
def cross_validation_knn(k_folds, hyperparameters, inputs, labels):
    best_hyperparam = 0
    best_accuracy = 0
    accuracies = np.zeros(len(hyperparameters))
    ########
    # TO DO:
    # preprocess for wrong data
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    for hp in hyperparameters:
        acc = 0
        fold_len = len(inputs) // k_folds
        # split inputs and labels into valid set and train set
        for i in range(k_folds):
            valid_inputs = inputs[i * fold_len : (i+1) * fold_len]
            valid_labels = labels[i * fold_len : (i+1) * fold_len]
            train_inputs = np.concatenate((inputs[: i * fold_len], inputs[(i+1) * fold_len:]))
            train_labels = np.concatenate((labels[: i * fold_len], labels[(i+1) * fold_len:]))
            acc += eval_knn(valid_inputs, valid_labels, train_inputs, train_labels, hp)
        # calculate avearge accuracy and update
        avg_accuracy = acc / k_folds
        accuracies[hp-1] = avg_accuracy
        if best_accuracy < avg_accuracy:
            best_hyperparam = hp
            best_accuracy = avg_accuracy

    ########
    return best_hyperparam, best_accuracy, accuracies


'''
This function plots the KNN accuracies for different # of neighbors (hyperparameters) based on cross validation

Inputs:
    **accuracies**: vector of accuracies for the corresponding hyperparameters (numpy array of floats)
    **hyperparams**: list of hyperparameters where each hyperparameter is a different # of neighbors (list of integers)
'''
def plot_knn_accuracies(accuracies, hyperparams):
    plt.plot(hyperparams, accuracies)
    plt.ylabel('accuracy')
    plt.xlabel('k neighbors')
    plt.show()


def main():
    # load data
    train_inputs, train_labels, test_inputs, test_labels = load_knn_data()
    print(train_inputs)
    
    # number of neighbors to be evaluated by cross validation
    hyperparams = range(1,31)
    k_folds = 10

    # use k-fold cross validation to find the best # of neighbors for KNN
    best_k_neighbors, best_accuracy, accuracies = cross_validation_knn(k_folds, hyperparams, train_inputs, train_labels)

    # plot results
    plot_knn_accuracies(accuracies, hyperparams)
    print('best # of neighbors k: ' + str(best_k_neighbors))
    print('best cross validation accuracy: ' + str(best_accuracy))

    # evaluate with best # of neighbors
    accuracy = eval_knn(test_inputs, test_labels, train_inputs, train_labels, best_k_neighbors)
    print('test accuracy: '+ str(accuracy))


if __name__ == "__main__":
    main()