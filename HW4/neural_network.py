import numpy as np


def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    Arguments:
        - input: input vector (N, in_features + 1) 
            WITH bias feature added as 1st col
        - p: parameter matrix (out_features, in_features + 1)
            WITH bias parameter added as 1st col (i.e. alpha / beta in the writeup)

    Returns:
        - output vector (N, out_features)
    """
    return np.dot(input, p.T)


def sigmoidForward(a):
    """
    Arguments:
        - a: input vector (N, dim)

    Returns:
        - output vector (N, dim)
    """
    return 1 / (1 + np.exp(-a))


def softmaxForward(b):
    """
    Arguments:
        - b: input vector (N, dim)

    Returns:
        - output vector (N, dim)
    """
    max_value = np.max(b)
    exp_b = np.exp(b - max_value)
    exp_sum = np.sum(exp_b)
    return exp_b / exp_sum


def crossEntropyForward(hot_y, y_hat):
    """
    Arguments:
        - hot_y: 1-hot encoding for true labels (N, K), where K is the # of classes
        - y_hat: (N, K) vector of probabilistic distribution for predicted label

    Returns:
        - average cross entropy loss for N data (scalar)
    """
    # return -np.mean(np.sum(hot_y * np.log(y_hat), axis=1))
    return np.mean(-np.sum(hot_y * np.log(y_hat + 1e-15), axis=1))


def NNForward(x, y, alpha, beta):
    """
    Arguments:
        - x: input vector (N, M+1)
            WITH bias feature added as 1st col
        - y: ground truth labels (N,)
        - alpha: alpha parameter matrix (D, M+1)
            WITH bias parameter added as 1st col
        - beta: beta parameter matrix (K, D+1)
            WITH bias parameter added as 1st col

    Returns:
        - a: 1st linear output (N, D)
        - z: sigmoid output WITH bias feature added as 1st col (N, D+1)
        - b: 2nd linear output (N, K)
        - y_hat: softmax output (N, K)
        - J: cross entropy loss (scalar)

    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    bias_feature = 1
    a = linearForward(x, alpha)
    z = np.insert(sigmoidForward(a), 0, bias_feature, axis=1)
    b = linearForward(z, beta)
    y_hat = softmaxForward(b)
    hot_y = np.zeros(y_hat.shape)
    for i in range(len(y)):
        hot_y[i, y[i]] = 1
    J = crossEntropyForward(hot_y, y_hat)
    return x, a, z, b, y_hat, J


def softmaxBackward(hot_y, y_hat):
    """
    Arguments:
        - hot_y: 1-hot encoding for true labels (N, K) where K is the # of classes
        - y_hat: (N, K) vector of probabilistic distribution for predicted label
    """
    return y_hat - hot_y


def linearBackward(prev, p, grad_curr):
    """
    Arguments:
        - prev: previous layer WITH bias feature
        - p: parameter matrix (alpha/beta) WITH bias parameter
        - grad_curr: gradients for current layer

    Returns:
        - grad_param: gradients for parameter matrix (i.e. alpha / beta)
            This should have the same shape as the parameter matrix.
        - grad_prevl: gradients for previous layer WITHOUT bias

    TIP: Check your dimensions.
    """
    # prev(z): (N, D+1), p(beta): (K, D+1), grad_curr(g_b): (N, K)
    # grad_param(g_beta): (K, D+1), grad_prevl(g_z): (N, D)

    # prev(x): (N, M+1), p(alpha): (D, M+1), grad_curr(g_a): (N, D)
    # grad_param(g_alpha): (D, M+1)
    grad_param = np.dot(grad_curr.T, prev)
    grad_prev = np.dot(grad_curr, p[:, 1:])
    return grad_param, grad_prev


def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """
    # curr(z): (N, D+1), grad_curr(g_z): (N, D)
    curr_wo_bias = curr[:, 1:]
    sigmoid_derivative = curr_wo_bias * (1 - curr_wo_bias)
    grad_prevl = grad_curr * sigmoid_derivative
    
    return grad_prevl


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    Arguments:
        - x: input vector (N, M+1)
        - y: ground truth labels (N,)
        - alpha: alpha parameter matrix (D, M+1)
            WITH bias parameter added as 1st col
        - beta: beta parameter matrix (K, D+1)
            WITH bias parameter added as 1st col
        - z: z as per writeup (N, D+1)
        - y_hat: (N, K) vector of probabilistic distribution for predicted label

    Returns:
        - g_alpha: gradients for alpha
        - g_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    """
    hot_y = np.zeros(y_hat.shape)
    for i in range(len(y)):
        hot_y[i, y[i]] = 1
    g_b = softmaxBackward(hot_y, y_hat)
    g_beta, g_z = linearBackward(z, beta, g_b)
    g_a = sigmoidBackward(z, g_z)
    g_alpha, g_x = linearBackward(x, alpha, g_a)
    return g_alpha, g_beta, g_b, g_z, g_a

def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    """
    Arguments:
        - tr_x: training data input (N_train, M)
        - tr_y: training labels (N_train, 1)
        - valid_x: validation data input (N_valid, M)
        - valid_y: validation labels (N_valid, 1)
        - hidden_units: Number of hidden units
        - num_epoch: Number of epochs
        - init_flag:
            - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
            - False: Initialize weights and bias to 0
        - learning_rate: Learning rate

    Returns:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    train_entropy = []
    valid_entropy = []
    N, M = tr_x.shape
    K = 10
    minibatch_size = 1
    # x: (N, M) -> (N, M+1)
    bias_value = 1
    tr_x = np.insert(tr_x, 0, bias_value, axis=1)
    valid_x = np.insert(valid_x, 0, bias_value, axis=1)
    # init parameters
    alpha_shape = (hidden_units, M + 1)
    beta_shape = (K, hidden_units + 1)
    alpha = np.random.uniform(-0.1, 0.1, size=alpha_shape) if init_flag else np.zeros(alpha_shape)
    beta = np.random.uniform(-0.1, 0.1, beta_shape) if init_flag else np.zeros(beta_shape)
    alpha[:, 0] = 0     # init bias
    beta[:, 0] = 0      # init bias
    
    for e in range(num_epoch):
        for idx in range(0, len(tr_x), minibatch_size):
            x = tr_x[idx:idx+minibatch_size, :]
            y = tr_y[idx:idx+minibatch_size]
            _x, a, z, b, y_hat, J = NNForward(x, y, alpha, beta)
            g_alpha, g_beta, _, _, _ = NNBackward(x, y, alpha, beta, z, y_hat)
            alpha = alpha - learning_rate * g_alpha
            beta = beta - learning_rate * g_beta
        _, _, _, _, _, tr_J = NNForward(tr_x, tr_y, alpha, beta)
        _, _, _, _, _, valid_J = NNForward(valid_x, valid_y, alpha, beta)
        train_entropy.append(tr_J)
        valid_entropy.append(valid_J)
    
    return alpha, beta, train_entropy, valid_entropy


def prediction(tr_x, tr_y, valid_x, valid_y, tr_alpha, tr_beta):
    """
    Arguments:
        - tr_x: training data input (N_train, M)
        - tr_y: training labels (N_train, 1)
        - valid_x: validation data input (N_valid, M)
        - valid_y: validation labels (N-valid, 1)
        - tr_alpha: alpha weights WITH bias
        - tr_beta: beta weights WITH bias

    Returns:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data
        - y_hat_valid: predicted labels for validation data
    """
    N_train, _ = tr_x.shape
    N_valid, _ = valid_x.shape
    tr_error_cnt = 0
    valid_error_cnt = 0
    y_hat_train = np.zeros((N_train, 1))
    y_hat_valid = np.zeros((N_valid, 1))
    # x: (N, M) -> (N, M+1)
    bias_value = 1
    tr_x = np.insert(tr_x, 0, bias_value, axis=1)
    valid_x = np.insert(valid_x, 0, bias_value, axis=1)

    for idx in range(len(tr_x)):
        x = tr_x[idx:idx+1, :]
        y = tr_y[idx:idx+1]
        _, _, _, _, y_hat, _ = NNForward(x, y, tr_alpha, tr_beta)
        l = np.argmax(y_hat[0])
        if l != y[0]:
            tr_error_cnt += 1
        y_hat_train[idx, 0] = l
    for idx in range(len(valid_x)):
        x = valid_x[idx:idx+1, :]
        y = valid_y[idx:idx+1]
        _, _, _, _, y_hat, _ = NNForward(x, y, tr_alpha, tr_beta)
        l = np.argmax(y_hat[0])
        if l != y[0]:
            valid_error_cnt += 1
        y_hat_valid[idx, 0] = l
    return tr_error_cnt / N_train, valid_error_cnt / N_valid, y_hat_train, y_hat_valid

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def train_and_valid(X_train, y_train, X_val, y_val, num_epoch, num_hidden, init_rand, learning_rate):
    """ Main function to train and validate your neural network implementation.

        X_train: Training input in N_train-x-M numpy nd array. Each value is binary, in {0,1}.
        y_train: Training labels in N_train-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        X_val: Validation input in N_val-x-M numpy nd array. Each value is binary, in {0,1}.
        y_val: Validation labels in N_val-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        num_epoch: Positive integer representing the number of epochs to train (i.e. number of
            loops through the training data).
        num_hidden: Positive integer representing the number of hidden units.
        init_flag: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
        learning_rate: Float value specifying the learning rate for SGD.

        RETURNS: a tuple of the following six objects, in order:
        loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        y_hat_train: A list of integers representing the predicted labels for training data
        y_hat_val: A list of integers representing the predicted labels for validation data
    """
    ### YOUR CODE HERE
    loss_per_epoch_train = []
    loss_per_epoch_val = []
    err_train = None
    err_val = None
    y_hat_train = None
    y_hat_val = None
    
    alpha, beta, loss_per_epoch_train, loss_per_epoch_val = SGD(X_train, y_train, X_val, y_val, num_hidden, num_epoch, init_rand, learning_rate)
    err_train, err_val, y_hat_train, y_hat_val = prediction(X_train, y_train, X_val, y_val, alpha, beta)

    return (loss_per_epoch_train, loss_per_epoch_val,
            err_train, err_val, y_hat_train, y_hat_val)
    
def main():
    X_train, y_train, X_valid, y_valid = load_data_medium()
    num_epoch = 100
    num_hidden = 20
    init_rand = True
    learning_rate = 0.01
    
    # load_dic = {'small': load_data_small, 'medium': load_data_medium, 'large': load_data_large}
    # X_train, y_train, X_valid, y_valid = load_dic[data_size]()
    (loss_per_epoch_train, loss_per_epoch_val,
     err_train, err_val,
     y_hat_train, y_hat_val) = train_and_valid(X_train, y_train, X_valid, y_valid, num_epoch, num_hidden, init_rand, learning_rate)
    
    print(f"loss_per_epoch_train: {loss_per_epoch_train}")
    print(f"loss_per_epoch_val: {loss_per_epoch_val}")
    print(f"err_train: {err_train}")
    print(f"err_val: {err_val}")
    print(f"y_hat_train: {y_hat_train.flatten()}")
    print(f"y_train: {y_train}")
    print(f"y_hat_val: {y_hat_val.flatten()}")
    print(f"y_valid: {y_valid}")
    
if __name__ == "__main__":
    main()