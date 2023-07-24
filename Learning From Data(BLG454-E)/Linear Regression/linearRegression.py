#Author: Muhammet Serdar NAZLI - 150210723
#Empty spaces are filled for the HW1 of BLG454E course.

import numpy as np

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class minMaxScaler():
    def __init__(self, x, min_limit=0, max_limit=1):
        self.x = x
        self.min = np.min(x, axis=0)
        self.max = np.max(x, axis=0)
        self.min_limit = min_limit
        self.max_limit = max_limit
  
    def transform(self, x):
        x_scaled = x

        
        x_std = (x-self.min) / (self.max- self.min)
        x_scaled = x_std * (self.max_limit - self.min_limit) + self.min_limit

        return x_scaled
    
    def inverse_transform(self, x_scaled):
        x = x_scaled
        

        #I took the inverse function of transform mathematically with pen and paper. This formula came out 
        #and it is working correctly.
        x = (x_scaled - self.min_limit) * (self.max - self.min) / (self.max_limit - self.min_limit) + self.min 
        
      
        return x



def leastSquares(x, y):
    w = np.zeros((x.shape[1]+1, 1))

    """Since least squares says w = (X.T * X)^(-1) * X.T * y, let's implement it.

    In order to find bias term, we are going to concatenate the x with 
    a vector that consist of ones. This will represent bias in our model.
    """
    bias_ = np.ones((x.shape[0],1))
    x = np.hstack((bias_,x))
    w = np.matmul(np.matmul( np.linalg.inv(np.matmul(x.T , x)), x.T),   y)

    return w

class gradientDescent():
    def __init__(self, x, y, w, lr, num_iters):
        self.x = np.c_[np.ones((x.shape[0], 1)), x]
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w
        self.weight_history = [self.w]
        self.cost_history = [np.sum(np.square(self.predict(self.x)-self.y))/self.x.shape[0]]
        
    def gradient(self):
        gradient = np.zeros_like(self.w)
        """
        Since loss function L(w,b) = 1/2m * sigma(f_{w,b}(x)-y)^2
        the partial derivative of L respect to w is:
        dL(w,b)/dw = 1/m * x^T @ (x @ w - y) where m is the number of instances.
        """
        
        gradient = (1/self.x.shape[0]) * np.matmul(self.x.T, np.matmul(self.x, self.w) - self.y)

        return gradient
    
    
    def fit(self,lr=None, n_iterations=None):
        k = 0
        if n_iterations is None:
            n_iterations = self.num_iters
        if lr != None and lr != "diminishing":
            self.lr = lr
        # k represents the number of iterations that we made.
        while(k < n_iterations):
            #Gradient part.
            self.w = self.w - self.lr * self.gradient()

            #Add the new weight to weight_history.
            self.weight_history.append(self.w)

            #cost = 1/(2*self.x.shape[0]) * np.sum((self.predict(self.x)-self.y) ** 2)
            cost = np.sum(np.square(self.predict(self.x)-self.y))/self.x.shape[0]

            #Add the cost to cost_history
            self.cost_history.append(cost)

            #If cost is smaller than the epsilon, no further gradient descending, breaking the loop.
            if cost <= self.epsilon:
                break
            
            #If lr is diminishing, updating the self.lr. 
            if lr == "diminishing":
                self.lr = 1/(k+1)
                
            k += 1

        return self.w, k
    
    def predict(self, x):
        
        y_pred = np.zeros_like(self.y)

        y_pred = x.dot(self.w)

        return y_pred
