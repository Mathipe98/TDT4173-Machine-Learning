import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:

    def __init__(self, hard=False):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.weights = None
        self.bias = None
        self.hard = hard

    def fit(self, X, y, epochs=400, learning_rate=0.1):
        """
        Estimates parameters for the classifier
        Args:
            X: (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y: (array<m>): a vector of floats containing
                m binary 0.0/1.0 labels
            epochs: (int) The number of times the main training loop will repeat
            learning_rate: (float) The learning rate that will adjust how quickly or slowly the weights and bias will
                            update
        """
        # If we are doing the hard version, i.e. dataset 2, then we will replace columns x0 and x1 with a new column
        # x2, s.t. x2 = x0^2 + x1^2 - 1. It's because the second set is circular, so we make the data linearly separable
        if self.hard:
            X_temp = X
            X_temp['x2'] = X_temp['x0'] ** 2 + X_temp['x1'] ** 2 - 1
            X = X_temp.drop(columns=['x0', 'x1'])
        num_features = len(X.columns)
        num_examples = len(X.index)
        # First we just normalize all the values in the dataframe
        X = normalize_dataframe(X)
        # Then we initialize weights and bias to be uniformly distributed between -0.5 and 0.5
        # (I did this in a neural network from scratch task, so I figured I might as well try it here as well)
        weights = np.random.uniform(low=-0.5, high=0.5, size=(num_features, 1))
        bias = np.random.uniform(low=-0.5, high=0.5)
        # We reshape the true y-values such that we can subtract it from the prediction vector that we get out from
        # the sigmoid function (basically to have shape (500, 1) instead of (500,) cuz numpy strict)
        y_true = np.array(y).reshape(num_examples, 1)

        # Start the training loop
        for epoch in range(epochs * num_examples):
            # First we calculate the product between each example, and the weights
            # (we do this vectorized to save runtime)
            # It will correspond to taking each of the x-values in X (all x0 and x1s) and multiplying them
            # by their individual weights in the weights-matrix we created earlier
            vector_result = np.dot(X, weights)
            # Then we calculate our predictions by feeding the vector result + the bias term into the sigmoid function
            y_pred = sigmoid(vector_result + bias)
            # We calculate how much we need to adjust the weight(s) by, and how much we adjust the bias by
            weight_adjustment, bias_adjustment = calculate_gradients(X, y_true, y_pred)
            # Finally update the parameters according to the calculated gradients (and learning rate)
            weights += learning_rate * weight_adjustment
            bias += learning_rate * bias_adjustment
        # updating self to save weights and bias
        self.weights = weights
        self.bias = bias

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        X = pd.DataFrame(X, columns=['x0', 'x1'])
        # If we're in the second dataset, then we again reduce the 2 columns to 1 single column
        if self.hard:
            X_temp = X
            X_temp['x2'] = X_temp['x0'] ** 2 + X_temp['x1'] ** 2 - 1
            X = X_temp.drop(columns=['x0', 'x1'])
        # Again we normalize the values
        X = normalize_dataframe(X)
        # We calculate the predictions by shoving everything into sigmoid (all hail sigmoid)
        predictions = sigmoid(np.dot(X, self.weights) + self.bias)
        # We return the array of predictions where we round to the nearest integer (either 0 or 1)
        return np.array([1 if prediction > 0.5 else 0 for prediction in predictions])


# --- Some utility functions

def normalize_dataframe(X):
    """
    This function will take in a dataframe object (which will just be an array of m x n dimensions) and normalize all
    values in it, such that all values lie in the range [0, 1]. This is something that is regularly done for neural
    networks, and as such I will do the same here. It will use min-max normalization
    Args:
        X: (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

    Returns:
        X: (array<m,n>): a matrix of floats in the range [0, 1] with
                m rows (#samples) and n columns (#features)
    """
    return (X - X.min()) / (X.max() - X.min())


def calculate_gradients(X, y_true, y_pred):
    """
    This function will calculate the gradients between the predicted outcome and the actual answers
    when training the model. This will happen when the regression model predicts the answers to each
    example in the training loop (in the epoch section), and the gradient will calculate the "average
    distance" away from the correct answer, and return the adjustment to the weights accordingly.
    This is in line with stochastic gradient descent, i.e. mathematically calculating the average
    distance away from the local (and hopefully global) minimum of the function we're trying to predict.
    Args:
        X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        y_true (array<m>): a vector of floats containing
                m binary 0.0/1.0 labels

        y_pred (array<m>): a vector of floats containing
                m binary 0.0/1.0 labels that are predicted in the self.fit function
    Returns:
        (tuple<2>): a tuple consisting of 2 values; the first being an array of shape <1, n> where n is the number of
        features in the training set, and the second being a scalar.
    """
    # First we get the number of training examples from the dataset
    num_examples = len(X.index)
    # We then use the gradient descent formula from the notes provided in the notebook
    # The following line corresponds to y - h_{theta}(x) (since we use the entire vectors, thus no ^(i))
    prediction_difference = y_true - y_pred
    # Then we multiply every training example with the difference between prediction and true value
    differences_x_examples = np.dot(X.T, prediction_difference)
    # Now we have calculated (y^(i) - h_{theta}(x^(i)) ) x_{j}^(i) for all i and all j
    # Finally we need to divide this by the number of examples since we do all the adjustments in one go
    # (rather than iterating through every example row and adjusting the weight(s) one example at a time)
    weight_adjustment = (1/num_examples) * differences_x_examples
    # We do the same for the bias (however just in one line since we only have 1 bias value)
    bias_adjustment = (1 / num_examples) * np.sum((y_true - y_pred))
    # Finally we return these adjustments to be used in the training-loop for updating the weights and bias
    return weight_adjustment, bias_adjustment


def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptable
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))
