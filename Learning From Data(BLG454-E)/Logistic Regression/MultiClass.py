############################################################################################
#               Implementation of MultiClass Logistic Regression.                          #
############################################################################################
#Muhammet Serdar NAZLI, 150210723

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time


def sigmoid(z):
    """
    Computes the sigmoid function element wise over a numpy array.
    :param z: A numpy array.
    :return: A numpy array of the same size of z.
    """

    return 1 / (1 + np.exp(-z)) #Sigmoid formula


def log_loss(X, Y, W, N):
    """
    Computes the log-loss function, and its gradient over a mini-batch of data.
    :param X: The feature matrix of size (N, F+1), where F is the number of features.
    :param Y: The label vector of size (N, 1).
    :param W: The weight vector of size (F+1, 1).
    :param N: The number of samples in the mini-batch.
    :return: Loss (a scalar) and gradient (a vector of size (F+1, 1)).
    """
  

    Y_hat = sigmoid(X.dot(W))       #Multiply the data points with the weights, then send it to sigmoid function.

    #The return is loss[0][0]. I could not understand why. In order to make this function work
    #I had to cover the output with square brackets two times. It works without any problem.
    loss =  np.array([[(-1/N) * np.sum(Y * np.log(Y_hat) + (1-Y) * np.log(1-Y_hat))]]) #L(y, p) = -1/N \sigma [y * log(p) + (1 - y) * log(1 - p)]
    grad =  (1/N) * X.T.dot(Y_hat - Y) #Partial derivative of the loss function with respect to W.

    return loss[0][0], grad


def visualizer(loss, accuracy, n_epochs):
    """
    Returns the plot of Training/Validation Loss and Accuracy.
    :param loss: A list defaultdict with 2 keys "train" and "val".
    :param accuracy: A list defaultdict with 2 keys "train" and "val".
    :param n_epochs: Number of Epochs during training.
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    x = np.arange(0, n_epochs, 1)
    axs[0].plot(x, loss['train'], 'b')
    axs[0].plot(x, loss['val'], 'r')
    axs[1].plot(x, accuracy['train'], 'b')
    axs[1].plot(x, accuracy['val'], 'r')

    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss value")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy value (in %)")

    axs[0].legend(['Training loss', 'Validation loss'])
    axs[1].legend(['Training accuracy', 'Validation accuracy'])


class OneVsAll:
    def __init__(self, x_train, y_train, x_test, y_test, alpha, beta, mb, n_class, F, n_epochs, info):
        """
        This is an implementation from scratch of Multi Class Logistic Regression using One vs All strategy,
        and Momentum with SGD optimizer.

        :param x_train: Vectorized training data.
        :param y_train: Label training vector.
        :param x_test: Vectorized testing data.
        :param y_test: Label test vector.
        :param alpha: The learning rate.
        :param beta: Momentum parameter.
        :param mb: Mini-batch size.
        :param n_class: Number of classes.
        :param F: Number of features.
        :param n_epochs: Number of Epochs.
        :param info: 1 to show training loss & accuracy over epochs, 0 otherwise.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.alpha = alpha
        self.beta = beta
        self.mb = mb 
        self.n_class = n_class
        self.F = F 
        self.n_epochs = n_epochs
        self.info = info

    def relabel(self, label):
        """
        This function takes a class, and relabels the training label vector into a binary class,
        it's used to apply One vs All strategy.

        :param label: The class to relabel.
        :return: A new binary label vector.
        """

        y = self.y_train.tolist()
        n = len(y)
        y_new = [1 if y[i] == label else 0 for i in range(n)]

        return np.array(y_new).reshape(-1, 1)

    def momentum(self, y_relab):
        """
        This function is an implementation of the momentum with SGD optimization algorithm, and it's
        used to find the optimal weight vector of the logistic regression algorithm.
        :param y_relab: A binary label vector.
        :return: A weight vector, and history of loss/accuracy over epochs.
        """


        # Initialize weights and velocity vectors
        W = np.zeros((self.F + 1, 1))
        V = np.zeros((self.F + 1, 1))

        # Store loss & accuracy values for plotting
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        # Split into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, y_relab, test_size=0.1, random_state=42)
        n_train = len(x_train)
        n_val = len(x_val)

  
        for _ in range(self.n_epochs):

            start = time.time()
            train_loss = 0
            # Compute the loss and gradient over mini-batches of the training set
            # I am not sure but, shouldn't the step size be self.mb instead of 1?
            for i in range(0, n_train - self.mb + 1):
                train_loss, grad =  log_loss(x_train[i:i+self.mb], y_train[i:i+self.mb], W, self.mb) #Indexing x_train and y_train to get batches than sending to log_loss function.
                V = self.beta * V + grad       #Velocity formula
                W = W - self.alpha * V         #Updating weights with velocity.

            # Compute the training accuracy
            train_proba = sigmoid(x_train.dot(W))   #Getting y^hat
            train_class = (train_proba >= 0.5)      #The values bigger than or equal to 0.5 will be 1, others will be 0.
            train_acc =  np.mean(train_class == y_train) * 100  #Calculating accuracy

            # Compute the loss & accuracy over the validation set
            y_hat =  sigmoid(x_val.dot(W))                   #y^hat
            val_loss, __ = log_loss(x_val, y_val, W, n_val)  #In order to get loss, using log_loss function.
            val_class = (y_hat >= 0.5)                       #The values bigger than or equal to 0.5 will be 1, others will be 0.
            val_acc = np.mean(val_class == y_val) * 100      #Calculating accuracy


            end = time.time()
            duration = round(end - start, 2)

            if self.info: print("Epoch: {} | Duration: {}s | Train loss: {} |"
                                " Train accuracy: {}% | Validation loss: {} | "
                                "Validation accuracy: {}%".format(_, duration,
                                round(train_loss, 5), train_acc, round(val_loss, 5), val_acc))

            # Append training & validation accuracy and loss values to a list for plotting
            loss['train'].append(train_loss)
            loss['val'].append(val_loss)
            accuracy['train'].append(train_acc)
            accuracy['val'].append(val_acc)

        return W, loss, accuracy

    def train(self):
        """
        This function trains the model using One-vs-All strategy, and returns a weight
        matrix, to be used during inference.
        :return: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        """


        weights = []
        loss, accuracy = 0, 0

        for i in range(1, self.n_class + 1):
            print("-" * 50 + " Processing class {} ".format(i) + "-" * 50 + "\n")

            y_relab = self.relabel(i) 
            W, loss, accuracy = self.momentum(y_relab) 
            weights.append(W)

        # Get the weights matrix as a numpy array
        weights = np.hstack(weights)


        return weights, loss, accuracy

    def test(self, weights):
        """
        This function is used to test the model over new testing data samples, using
        the weights matrix obtained after training.
        :param weights: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        :return:
        """

        # Getting the probabilities matrix for each class over the testing set (ntest, n_class)
        proba = sigmoid(self.x_test.dot(weights))

        # For each sample in the probabilities matrix, we get the index of the maximum probability
        y_hat = np.argmax(proba, axis=1) + 1

        # Computing the test accuracy
        test_acc = np.mean(y_hat == self.y_test) * 100

        print("-" * 50 + "\n Test accuracy is {}%".format(test_acc))

