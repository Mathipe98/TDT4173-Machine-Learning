import numpy as np
import pandas as pd


# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class Node:
    """
    Helper class Node to represent a node in a decision tree.
    This object will have the following values:
        - Attribute: value representing the attribute that is being used to split this node
        - Branches: dictionary to keep track of child nodes. Key is the value of the Node-attribute,
                    (for example: if attribute = "Sex", then the key will either be "male" or "female")
                    and value is another Node-object
    """

    def __init__(self):
        # Attribute (i.e. Outlook, temp, etc.) that corresponds to this node in the tree
        self.attribute = None
        # Next node (if any) present in the tree
        self.next = None
        # 

    def add_branch(self, value, node):
        self.branches[value] = node

    def __str__(self):
        return self.attribute


class DecisionTree:

    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        # Variable to keep track of the attribute we want to predict
        self.goal = None
        # Keep track of the global entropy for the entire set (to use in info gain formula)
        self.total_entropy = None

    def calculate_remainder(self, attribute, examples):
        # Keep a running track of the result for summation
        result = 0
        # Iterate through every possible state for the attribute, and get p_k and n_k
        for k in self.training_set[attribute].unique():
            # Due to implementation, the last element is often NaN. So we just skip this
            if pd.isnull(k):
                continue
            # Default values are 0 in case examples don't exist
            p_k = 0
            n_k = 0
            # Count positive and negative outcomes
            resulting_example_count = examples[examples[attribute] == k].groupby(self.goal)[attribute].count()
            # Iterate through indexes in case examples for this don't exist
            for i in resulting_example_count.index:
                if i == 0:
                    n_k = resulting_example_count[0]
                if i == 1:
                    p_k = resulting_example_count[1]
            # If there are no negative nor positive examples, then we will add 0 anyway, so we just skip
            if p_k == 0 and n_k == 0:
                continue
            # Else add the values in accordance with the summation formula
            r1 = (p_k + n_k) / (self.p + self.n)
            r2 = self.calculate_boolean_entropy((p_k / (p_k + n_k)))
            result += r1 * r2
        return result

    def calculate_boolean_entropy(self, q):
        if q == 0 or q == 1:
            # return -(0 + (1-0) log(1)) = 0
            return 0
        # return B(q)
        r1 = q * np.log2(q, 2)
        r2 = (1 - q) * np.log2(1 - q, 2)
        return -(r1 + r2)

    def get_plurality_value(self, examples):
        # Examples will be a DataFrame with set attributes, for example ('Outlook' == 'Sunny') & ('Wind' == 'Weak')
        # We therefore just access the goal-column of these combined attributes, and count how many there are
        return 1 if examples[self.goal].value_counts()[1] > examples[self.goal].value_counts()[0] else 0

    def fit(self, X, y, attributes=None):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
            attributes: list of attributes from the dataset
        """
        # TODO: Implement
        # First we calculate the entropy of the entire set by value_counts, which is an array containing the number of
        # positive and negative classifications for the target attribute (which in case 1 is 'Play Tennis')
        self.total_entropy = entropy(y.value_counts())
        # We then set the attribute we want to predict (goal)
        self.goal = y.name
        # If there are only positive ('Yes') or negative ('No') attributes left, then return the current classification
        classifications = X[self.goal].value_counts()
        if len(classifications) == 1:
            return classifications.index[0]


    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement 
        raise NotImplementedError()

    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        raise NotImplementedError()


# --- Some utility functions 

def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a length k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))


if __name__ == '__main__':
    test_model = DecisionTree()
    train_set = pd.read_csv('data_1.csv')
    X = train_set.drop(columns=['Play Tennis'])
    y = train_set['Play Tennis']
    test_model.fit(X, y)
