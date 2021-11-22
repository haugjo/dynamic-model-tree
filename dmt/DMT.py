import numpy as np
from dmt.Node import Node


class DynamicModelTree:
    def __init__(self, n_classes=2, learning_rate=0.05, epsilon=10e-8, n_saved_candidates=100, p_replaceable_candidates=0.5, cat_features=None):
        """ Dynamic Model Tree
        A dynamic adaptation of Model Trees for more flexible and interpretable classification in data streams.

        Parameters
        ----------
        n_classes - (int) number of classes (required to choose between logit and multinomial logit weak learners
        learning_rate - (float) learning rate for gradient updates of simple models
        epsilon - (float) epsilon threshold required before attempting to split or prune (based on AIC)
        n_saved_candidates  - (int) no. of candidates per node for which we save statistics
        p_replaceable_candidates - (float) max. percent of candidates that may be replaced by new/better candidates per iteration
        cat_features - (list) indices of categorical variables
        """
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_saved_candidates = n_saved_candidates
        self.p_replaceable_candidates = p_replaceable_candidates

        if cat_features is None:
            self.cat_features = []
        else:
            self.cat_features = cat_features

        self.root = None  # root node

    def partial_fit(self, X, y):
        """ Partial fit function
        Fit the model tree given the current batch of observations X and targets y

        Parameters
        ----------
        X - (np.array) vector/matrix of observations
        y - (np.array) vector of target labels
        """
        if self.root is None:  # Create root node
            self.root = Node(n_classes=self.n_classes,
                             n_features=X.shape[1],
                             cat_features=self.cat_features,
                             learning_rate=self.learning_rate,
                             epsilon=self.epsilon,
                             n_saved_candidates=self.n_saved_candidates,
                             p_replaceable_candidates=self.p_replaceable_candidates)

        # Start the recursive update of nodes at root (id=0)
        self.root.update(X, y)

    def predict(self, X):
        """ Predict class label
        Return the predicted labels for the given (batch of) observations

        Parameters
        ----------
        X - (np.array) vector/matrix of observations

        Returns
        -------
        y_pred - (np.array) predicted labels per observation
        """
        if self.root is None:
            raise Exception('Tree must be trained first, before calling predict()')

        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.root.predict_observation(x=X[i, :])[0])

        return np.asarray(y_pred)

    def predict_proba(self, X):
        """ Predict class probabilities
        Return the predicted class probabilities for the given (batch of) observations

        Parameters
        ----------
        X - (np.array) vector/matrix of observations

        Returns
        -------
        y_pred_prob - (np.array) predicted class probabilities per observation
        """
        if self.root is None:
            raise Exception('Tree must be trained, before calling first predict()')

        y_prob = []
        for i in range(X.shape[0]):
            y_prob.append(self.root.predict_observation(x=X[i, :], get_prob=True).flatten())

        return np.asarray(y_prob)

    def n_nodes(self):
        """ Return the number of nodes, leaves and the depth of the tree

        Returns
        -------
        n_total - (int) total number of nodes
        n_leaf - (int) total number of leaves
        depth - (int) depth of the tree
        """
        return self._add_up_node_count(n_total=0, n_leaf=0, depth=0, node=self.root)

    def _add_up_node_count(self, n_total, n_leaf, depth, node):
        """ Increment node counts

        Parameters
        ----------
        n_total - (int) current total no. of nodes
        n_leaf - (int) current no. of leaves
        depth - (int) current depth
        node - (Node) current node

        Returns
        -------
        n_total, n_leaf, depth
        """
        if node.is_leaf:
            return n_total + 1, n_leaf + 1, depth + 1
        else:
            n_total += 1
            depth += 1
            max_depth = depth
            for child in node.children:
                n_total, n_leaf, c_depth = self._add_up_node_count(n_total, n_leaf, depth, child)
                if c_depth > max_depth:  # return max depth returned by a child
                    max_depth = c_depth

            return n_total, n_leaf, max_depth
