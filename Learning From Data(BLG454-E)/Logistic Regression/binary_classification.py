import numpy as np
#Muhammet Serdar NAZLI: 150210723
class LogisticRegression:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor assumes a x_train matrix in which each column contains an instance.
        Vector y_train contains one integer for each instance, indicating the instance's label. 
        
        Constructor initializes the weights W and B, alpha, and a one-vector Y containing the labels
        of the training set. Here we assume there are 10 labels in the dataset. 
        """
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._m = x_train.shape[1]
        
        self._W = np.random.randn(10, 784) * 0.01
        self._B = np.zeros((10, 1))
        self._Y = np.zeros((10, self._m))
        self._alpha = 0.05

        for index, value in enumerate(self._y_train):
            self._Y[value][index] = 1
            
    def sigmoid(self, Z):
        """
            Args:
            - Z (numpy.ndarray): The input array.

            Returns:
            - numpy.ndarray: An array of the same shape as Z, where each element is the sigmoid of the corresponding element in Z.
            
        """
        
        ##############################################################################
        #Computes the sigmoid value for all values in vector Z


        #Since the sigmoid function is 1/(1+e^(-x)) 
        sigmoid_val = 1 / (1 + np.exp(-Z))
        

        return sigmoid_val

    def derivative_sigmoid(self, A):
        """
            Args:
            - A (numpy.ndarray): The input array.

            Returns:
            - numpy.ndarray: An array of the same shape as A, where each element is the derivative of the sigmoid function
            for the corresponding element in A.
        """


        #d((1+e^(-x))^(-1))/dx = e^(-x) / (1+e^(-x))^2, final expression can be written as 
        # (1/(1+e^(-x))) * (1 - 1/(1+e^(-x))) = sigmoid(x) * (1-sigmoid(x)) 
        sigmoid_val = self.sigmoid(A)
        deriv_sigmoid_val = sigmoid_val * (1-sigmoid_val)


        return deriv_sigmoid_val

    def h_theta(self, X):
        """
            Args:
            - X (numpy.ndarray): The input feature matrix.

            Returns:
            - numpy.ndarray: A column vector of predicted values obtained
        """
        z = self._W.dot(X) + self._B
        h_theta = self.sigmoid(z)


        return h_theta
    
    def return_weights_of_digit(self, digit):
        """
            Args:
            - digit (int): The digit for which the weights are to be returned.

            Returns:
            - numpy.ndarray: A row vector of weights from the weights matrix corresponding to the given digit.
        """


        #Getting the wanted row and all columns in that row.
        weights_of_digits = self._W[digit,:]
        

        return weights_of_digits
    
    def train_mse_loss(self, iterations):
        """
        Performs a number of iterations of gradient descend equals to the parameter passed as input.
        
        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []
        
        for i in range(iterations):
            A = self.h_theta(self._x_train)
            pure_error = A - self._Y
            self._W -= self._alpha * (1/self._m) * A.dot(self.derivative_sigmoid(self._x_train).T)
            self._B -= self._alpha * (2/self._m) * np.sum(pure_error * self.derivative_sigmoid(A))




            if i % 100 == 0:
                classified_correctly = np.sum(np.argmax(A, axis=0) == self._y_train)

                percentage_classified_correctly = (classified_correctly / self._m)*100
                classified_correctly_train_list.append(percentage_classified_correctly)
                
                Y_hat_test = self.h_theta(self._x_test)
                test_correct = np.sum(Y_hat_test == self._y_test)
                classified_correctly_test_list.append((test_correct)/len(self._y_test) * 100)
                
                print('Accuracy train data: %.2f' % percentage_classified_correctly)

        return classified_correctly_train_list, classified_correctly_test_list


    def train_cross_entropy_loss(self, iterations):
        """
        Performs a number of iterations of gradient descend equals to the parameter passed as input.
        
        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """

        classified_correctly_train_list_ce = []
        classified_correctly_test_list_ce = []
        
        for i in range(iterations):
            A = self.h_theta(self._x_train)

            pure_error = A - self._Y

            self._W -= self._alpha * (1/self._m) *pure_error.dot(self._x_train.T)
            self._B -= self._alpha * (1/self._m) * np.sum(pure_error)

            if i % 100 == 0:
                classified_correctly = np.sum(np.argmax(A, axis=0) == self._y_train)
                percentage_classified_correctly_ce = (classified_correctly / self._m)*100
                classified_correctly_train_list_ce.append(percentage_classified_correctly_ce)
                
                Y_hat_test = np.argmax(self.h_theta(self._x_test), axis=0)
                test_correct = np.sum(Y_hat_test == self._y_test)
                classified_correctly_test_list_ce.append((test_correct)/len(self._y_test) * 100)
                
                print('Accuracy train data: %.2f' % percentage_classified_correctly_ce)
        return classified_correctly_train_list_ce, classified_correctly_test_list_ce




