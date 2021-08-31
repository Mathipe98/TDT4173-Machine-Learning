import pandas as pd
import math
import numpy as np
import pprint

# All of this code is basically just a re-implementation of my DecisionTree-algorithm from TDT4171.
# It's not very streamlined, but it works, so I just saved time from having to implement it again


class Node:
    """
    Helper class Node to represent a node in a decision tree.
    This object will have the following values:
        - Attribute: value representing the attribute that is being used to split this node
        - Branches: dictionary to keep track of child nodes. Key is the value of the Node-attribute,
                    (for example: if attribute = "Wind", then the key will either be "Weak" or "Strong")
                    and value is either another Node-object, or an answer to the prediction ('Yes'/'No').
                    It's not best practice to have several different types of objects as values in
                    the same dictionary, but it works for my purposes here.
    """

    def __init__(self, attribute):
        # Attribute that corresponds to this node in the tree
        self.attribute = attribute
        # Node to keep track of branches. Key is v_k, value is a node (subtree) or a string
        self.branches = {}

    def add_branch(self, value, node):
        self.branches[value] = node

    def __str__(self):
        return self.attribute


class DecisionTree:

    def __init__(self):
        # Assign stuff that is to be used in the learning-algorithm
        self.training_set = None
        self.goal = None
        # Have a tree-result that will be used in algorithms for prediction and drawing (visualization)
        self.tree_result = None
        # All attributes that are to be considered
        self.attributes = None
        # Number of positive outcomes
        self.p = None
        # Number of negative outcomes
        self.n = None
        # Fixed boolean entropy for the total amount of positive and negative outcomes
        self.B = None

    def fit(self, X, y):
        # Here I basically adjust the data to fit to my previously-coded algorithm from TDT4171
        examples = X
        # Now if the data has zodiac signs, we ignore this because we are men and women of science, and we do not speak
        # in bullshit
        if 'Founder Zodiac' in examples.columns:
            examples.drop(columns=['Founder Zodiac'], inplace=True)
        # Create a list of the attributes we will train on (excluding the attribute we will predict)
        attributes = examples.columns
        # In my implementation, the prediction-column and the rest of the dataset need to be joined
        examples[y.name] = y
        # Here I just use fit as an init function
        self.training_set = examples
        self.goal = y.name
        self.attributes = attributes
        self.p = self.training_set[self.goal].value_counts()[1]
        self.n = self.training_set[self.goal].value_counts()[0]
        self.B = self.calculate_boolean_entropy(self.p / (self.p + self.n))
        self.decision_tree_learning(examples, attributes)

    def calculate_boolean_entropy(self, q):
        if q == 0 or q == 1:
            return 0
        # return B(q)
        r1 = q * math.log(q, 2)
        r2 = (1 - q) * math.log(1 - q, 2)
        return -(r1 + r2)

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

    def get_plurality_value(self, examples):
        # Examples will be a DataFrame with set attributes, for example ('Temp' == 'Hot') & ('Wind' == 'Weak')
        # We therefore pick the result that has the most amount of outcomes (either positive or negative)
        # by grouping the dataframe by outcomes, and then choosing the outcome that maximizes this amount
        return examples[self.goal].value_counts().index[np.argmax(examples[self.goal].value_counts())]

    def argmax(self, attributes, examples):
        # Assign the first attribute
        A = attributes[0]
        current_gain = self.B - self.calculate_remainder(A, examples)
        # Check if other attributes are better to start with by greedy information gain
        for current_attribute in attributes[1:]:
            test_gain = self.B - self.calculate_remainder(current_attribute, examples)
            # If current attribute gains more information than the one we had, then replace it
            if test_gain > current_gain:
                A = current_attribute
                current_gain = test_gain
        # Return the optimal attribute
        return A

    def decision_tree_learning(self, examples=None, attributes=None, parent_examples=None):
        # Assign starting examples (the entire dataset) if this is the first iteration
        if examples is None:
            examples = self.training_set
        # Assign attributes to split on. If use_cont is true, then we include continuous attributes
        if attributes is None:
            attributes = self.attributes
        # If there are no classification for the current combination of attributes, then return the most
        # common classification of the parent
        if examples.empty:
            return self.get_plurality_value(parent_examples)
        # Get a count of how many classifications there are for the current examples
        classifications = examples[self.goal].value_counts()
        # If there's only 1, i.e. every row in this example-set has the same outcome, then just return it
        if len(classifications) == 1:
            return classifications.index[0]
        # If there are no more attributes to query on, then just return whichever classification is more common
        if len(attributes) == 0:
            return self.get_plurality_value(examples)
        # A is the attribute that maximizes information gain
        A = self.argmax(attributes, examples)
        # Create the current node
        tree = Node(A)
        # Create a list of attributes that we will check for splitting
        current_attributes = [a for a in attributes if a is not A]
        # Iterate through the values for the attribute (Example: for "Humidity", v_k = "High" and v_k = "Normal")
        for v_k in self.training_set[A].unique():
            # New set of examples are examples where column-values match the attribute value v_k
            exs = examples[examples[A] == v_k]
            # Create a subtree with this set of examples
            subtree = self.decision_tree_learning(exs, current_attributes, examples)
            # Add the subtree to the current tree-node, making subtree a child of tree
            tree.add_branch(v_k, subtree)
        # Set the tree_result to the tree-node. Will become the root-node in the final iteration, which
        # is why it is added in the first place. Could be done in a better way, but works fine in our case.
        self.tree_result = tree
        return tree

    def predict_result(self, test_example, current_node):
        # Iterate through each child-node as these nodes determine which
        # attribute we consider and what value it has
        for attribute_value, child_node in current_node.branches.items():
            for attribute, value in test_example.items():
                # If it's not the same attribute in the column, then skip the column
                if attribute != current_node.attribute:
                    continue
                # We already know whether it's the same attribute, so now we check if it's the same value
                if value == attribute_value:
                    # If the child_node is not a Node-object, then the child_node is the classification value
                    if not isinstance(child_node, Node):
                        return child_node
                    # If we come here, then we have another level of nodes to investigate.
                    # Return the same check, but with a node on the level below
                    return self.predict_result(test_example, current_node.branches[attribute_value])

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
        # Create a list containing the results
        result_list = []
        # Iterate through each row in the test set
        for index, test_example in X.iterrows():
            # Predict the result using the tree from the learning-algorithm
            result = self.predict_result(test_example, self.tree_result)
            # Add the result to the resulting list
            result_list.append(result)
        # Return the results as a numpy array to work with the accuracy function used in the notebook
        return np.array(result_list)

    def get_rules(self, current_node=None, final_result_list=None, current_tuple_list=None):
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
        # If this is the first iteration, then we create the necessary variables
        if final_result_list is None and current_tuple_list is None and current_node is None:
            final_result_list = []
            current_tuple_list = []
            current_node = self.tree_result
        # Iterate through the attributes and the corresponding values in the branches dict of the root node
        for attribute_value, child in current_node.branches.items():
            # Add the current tuple to the list of tuples a layer down (this is in order to not change the
            # list of tuples on the layers above)
            next_tuple_list = current_tuple_list.copy()
            next_tuple_list.append((current_node.attribute, attribute_value))
            # If the child is a string (i.e. not a Node object), then that branch is finished, and we return it
            if not isinstance(child, Node):
                # The final tuple will contain a list of tuples, and a string (i.e. the prediction)
                end_tuple = (next_tuple_list, child)
                # We add this final tuple to the final resulting list
                final_result_list.append(end_tuple)
            else:
                # If the child is another node, then call the function recursively
                self.get_rules(child, final_result_list, next_tuple_list)
        return final_result_list


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


def problem_1():
    data_1 = pd.read_csv('data_1.csv')
    X = data_1.drop(columns=['Play Tennis'])
    y = data_1['Play Tennis']
    model_1 = DecisionTree()
    model_1.fit(X, y)
    rules = model_1.get_rules()
    pprint.pprint(rules)

def problem_2():
    data_2 = pd.read_csv('data_2.csv')
    data_2_train = data_2.query('Split == "train"')
    data_2_valid = data_2.query('Split == "valid"')
    data_2_test = data_2.query('Split == "test"')
    X_train, y_train = data_2_train.drop(columns=['Outcome', 'Split']), data_2_train.Outcome
    X_valid, y_valid = data_2_valid.drop(columns=['Outcome', 'Split']), data_2_valid.Outcome
    X_test, y_test = data_2_test.drop(columns=['Outcome', 'Split']), data_2_test.Outcome

    # Fit model (TO TRAIN SET ONLY)
    model_2 = DecisionTree()  # <-- Feel free to add hyperparameters
    model_2.fit(X_train, y_train)

    print(f'Train: {accuracy(y_train, model_2.predict(X_train)) * 100 :.1f}%')
    print(f'Valid: {accuracy(y_valid, model_2.predict(X_valid)) * 100 :.1f}%')
    print(f'Test: {accuracy(y_test, model_2.predict(X_test)) * 100 :.1f}%')




if __name__ == '__main__':
    problem_1()
    problem_2()
