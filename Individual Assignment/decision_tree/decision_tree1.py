import pandas as pd
import math
from graphviz import Digraph


class Node:
    """
    Helper class Node to represent a node in a decision tree.
    This object will have the following values:
        - Attribute: value representing the attribute that is being used to split this node
        - Branches: dictionary to keep track of child nodes. Key is the value of the Node-attribute,
                    (for example: if attribute = "Sex", then the key will either be "male" or "female")
                    and value is either another Node-object, or a number.
                    It's not best practice to have several different types of objects as values in
                    the same dictionary, but it works for our purposes here.
                    For future assessments, one should consider splitting this into different datastructures.
        - Identifier: Unique identifier that is set when drawing the tree using graphviz.
    """

    def __init__(self, attribute):
        # Attribute that corresponds to this node in the tree
        self.attribute = attribute
        # Node to keep track of branches. Key is v_k, value is a node (subtree)
        self.branches = {}
        # Identifier for graphviz
        self.identifier = None

    def add_branch(self, value, node):
        self.branches[value] = node

    def set_identifier(self, identifier):
        self.identifier = identifier

    def __str__(self):
        return self.attribute


class Decision_Tree:

    def __init__(self,
                 training_set=None,
                 test_set=None,
                 goal=None,
                 attributes=None):
        # Assign stuff that is to be used in the learning-algorithm
        # self.test_set = pd.read_csv('data_1.csv') if test_set is None else test_set
        self.training_set = pd.read_csv('data_1.csv') if training_set is None else training_set
        self.goal = goal
        # Have a tree-result that will be used in algorithms for prediction and drawing (visualization)
        self.tree_result = None
        # All attributes that are to be considered
        self.attributes = attributes
        # Number of positive outcomes
        self.p = self.training_set[self.goal].value_counts()[1]
        # Number of negative outcomes
        self.n = self.training_set[self.goal].value_counts()[0]
        # Fixed boolean entropy for the total amount of positive and negative outcomes
        self.B = self.calculate_boolean_entropy(self.p / (self.p + self.n))

    def calculate_boolean_entropy(self, q):
        if q == 0 or q == 1:
            # return -(0 + (1-0) log(1)) = 0
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
        # Examples will be a DataFrame with set attributes, for example ('Sex' == 'male') & ('SibSp' == 1)
        # We therefore just access the goal-column of these combined attributes, and count how many there are
        return 1 if examples[self.goal].value_counts()[1] > examples[self.goal].value_counts()[0] else 0

    def argmax(self, attributes, examples):
        # Assign the first attribute
        A = attributes[0]
        current_gain = self.B - self.calculate_remainder(A, examples)
        # Check if other attributes are better to start with
        for current_attribute in attributes[1:]:
            test_gain = self.B - self.calculate_remainder(current_attribute, examples)
            # If current attribute gains more information than the one we had, then replace it
            if test_gain > current_gain:
                A = current_attribute
        # Return the optimal attribute
        return A

    def fit(self, X, y):
        # Here I basically adjust the data to fit to my previously-coded algorithm from TDT4171
        examples = X
        # Assume the attribute we want to predict always is the last one
        attributes = X.columns[:-1]
        self.decision_tree_learning()

    def decision_tree_learning(self, examples=None, attributes=None, parent_examples=None, use_cont=False):
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
        if not attributes:
            return self.get_plurality_value(examples)
        # A is the attribute that maximizes information gain (in our case)
        A = self.argmax(attributes, examples)
        # Create the current node
        tree = Node(A)
        # Create a list of attributes that we will check for splitting
        current_attributes = [a for a in attributes if a is not A]
        # Iterate through the values for the attribute (Example: for "Sex", v_k = "male" and v_k = "female")
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

    def calculate_prediction_accuracy(self):
        # Check if the learning-algorithm has been called. If not, then just call it
        if self.tree_result is None:
            self.decision_tree_learning()
        # Keep track of right and wrong answers
        correct = 0
        wrong = 0
        # Iterate through each row in the test set
        for index, test_example in self.test_set.iterrows():
            # Predict the result using the tree from the learning-algorithm
            result = self.predict_result(test_example, self.tree_result)
            # Check if our prediction matches the actual answer, and increment the counters individually
            if result == test_example[self.goal]:
                correct += 1
            else:
                wrong += 1
        # Return a list of the right and wrong answers
        return [correct, wrong]


if __name__ == '__main__':
    df = pd.read_csv('data_1.csv')
    examples = df.columns[:-1]
    goal = df.columns[-1]
    
