from abc import ABCMeta
import numpy as np
import math
import copy
from sklearn.linear_model import SGDClassifier

RNG = np.random.default_rng(0)  # set default random state for reproducibility


class Node(metaclass=ABCMeta):
    def __init__(self, n_classes, n_features, cat_features, learning_rate, epsilon, n_saved_candidates, p_replaceable_candidates):
        """ Node of the Dynamic Model Tree

        Parameters
        ----------
        n_classes - (int) number of classes
        n_features - (int) number of features
        cat_features - (list) indices of categorical variables
        learning_rate - (float) learning rate for gradient updates
        epsilon - (float) epsilon threshold required before attempting to split or prune (based on AIC)
        n_saved_candidates - (int) max number of candidates for which we save statistics
        p_replaceable_candidates - (float) max percentage of candidates to be replaced by new candidates per iteration
        """
        self.n_classes = n_classes
        self.n_features = n_features
        self.cat_features = cat_features
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_saved_candidates = n_saved_candidates
        self.p_replaceable_candidates = p_replaceable_candidates

        self.linear_model = SGDClassifier(loss='log', penalty='l2', alpha=0, eta0=learning_rate, learning_rate='constant', random_state=0)
        self.counts = 0
        self.log_likelihoods = np.zeros(n_classes)

        self.counts_left = dict()               # number of observations in the left child per split candidate
        self.log_likelihoods_left = dict()      # likelihoods for left child per split candidate
        self.gradients_left = dict()            # gradients for left child per split candidate
        self.counts_right = dict()              # number of observations in the right child per split candidate
        self.log_likelihoods_right = dict()     # likelihoods for left child per split candidate
        self.gradients_right = dict()           # gradients for right child per split candidate

        self.children = []      # array of child nodes
        self.split = None       # split feature/value combination
        self.is_leaf = True     # indicate whether node is a leaf

    def update(self, X, y):
        """ Update the node and all descendants
        Update the parameters of the weak model at the given node.
        If the node is an inner node, we may attempt to split on a different feature or to replace the inner node
        by a leaf and drop all children. If the node is a leaf node, we may attempt to split.
        We evoke the function recursively for all children.

        Parameters
        ----------
        X - (np.array) vector/matrix of observations
        y - (np.array) vector of target labels
        """
        # Update the simple/linear model
        log_likelihood_X, gradient_X = self._update_linear_model(X, y)

        # Update/add/replace the statistics of candidate splits
        candidate_aic = self._add_and_replace_candidates(X, log_likelihood_X, gradient_X)

        if not self.is_leaf:
            if self.split[0] in self.cat_features:
                left_idx = X[:, self.split[0]] == self.split[1]
                right_idx = X[:, self.split[0]] != self.split[1]
            else:
                left_idx = X[:, self.split[0]] <= self.split[1]
                right_idx = X[:, self.split[0]] > self.split[1]

            X_left, X_right, y_left, y_right = X[left_idx], X[right_idx], y[left_idx], y[right_idx]  # split data

            if len(y_left) > 0:
                self.children[0].update(X=X_left, y=y_left)

            if len(y_right) > 0:
                self.children[1].update(X=X_right, y=y_right)

            do_split, top_split = self._check_for_split(candidate_aic=candidate_aic)

            if not do_split:
                # Replace inner node by leaf
                self.children = []
                self.split = None
                self.is_leaf = True
            elif do_split and self.split != top_split:
                # Replace current split with new top split
                self._make_inner_node(split=top_split)
        else:
            do_split, top_split = self._check_for_split(candidate_aic=candidate_aic)

            if do_split:
                # Replace leaf node by an inner node
                self._make_inner_node(split=top_split)

    def _make_inner_node(self, split):
        """ Make current node an inner node

        Parameters
        ----------
        split - (tuple) feature/value pair of the current optimal split
        """
        self.split = split
        self.children = []
        self.is_leaf = False

        # Create new children
        for i in range(2):
            self.children.append(Node(n_classes=self.n_classes,
                                      n_features=self.n_features,
                                      cat_features=self.cat_features,
                                      learning_rate=self.learning_rate,
                                      epsilon=self.epsilon,
                                      n_saved_candidates=self.n_saved_candidates,
                                      p_replaceable_candidates=self.p_replaceable_candidates))

        # Set statistics of LEFT child (as we dynamically replace candidates, we need to scale the saved statistics to
        # the same number of observations as the current node has observed).
        self.children[0].counts = round(self.counts_left[split] * self.counts / (self.counts_left[split] + self.counts_right[split]))
        relative_frac = np.divide(self.log_likelihoods_left[split],
                                  self.log_likelihoods_left[split] + self.log_likelihoods_right[split],
                                  out=np.zeros_like(self.log_likelihoods_left[split]),
                                  where=(self.log_likelihoods_left[split] + self.log_likelihoods_right[split]) != 0)
        self.children[0].log_likelihoods = self.log_likelihoods * relative_frac

        self.children[0].linear_model = copy.deepcopy(self.linear_model)
        if self.counts_left[split] > 0:  # Apply one gradient update to the parent parameters and assign to child model
            if self.n_classes == 2:
                self.children[0].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_) \
                                                      + self.learning_rate / self.counts_left[split] * self.gradients_left[split][0]  # use any gradient for update
            else:
                self.children[0].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_) \
                                                      + self.learning_rate / self.counts_left[split] * self.gradients_left[split]
        else:
            self.children[0].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_)

        # Set statistics of RIGHT child (as we dynamically replace candidates, we need to scale the saved statistics to
        # the same number of observations as the current node has observed).
        self.children[1].counts = self.counts - self.children[0].counts
        self.children[1].log_likelihoods = self.log_likelihoods - self.children[0].log_likelihoods

        self.children[1].linear_model = copy.deepcopy(self.linear_model)
        if self.counts_right[split] > 0:  # Apply one gradient update to the parent parameters and assign to child model
            if self.n_classes == 2:
                self.children[1].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_) \
                                                      + self.learning_rate / self.counts_right[split] * self.gradients_right[split][0]
            else:
                self.children[1].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_) \
                                                      + self.learning_rate / self.counts_right[split] * self.gradients_right[split]
        else:
            self.children[1].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_)

    def _update_linear_model(self, X, y):
        """ Update Simple Model

        Update the simple model via gradient ascent on the neg. log-likelihood loss.
        Afterwards, compute and store current likelihoods and gradients

        Parameters
        ----------
        X - (np.array) vector/matrix of observations
        y - (np.array) vector of target labels

        Returns
        -------
        log_likelihood_X - (np.array) log likelihoods regarding each observation
        gradient_X - (dict) gradients regarding the parameters of each class
        """
        self.counts += X.shape[0]
        self.linear_model.partial_fit(X, y, classes=np.arange(self.n_classes))

        log_prob_X = self.linear_model.predict_log_proba(X)
        log_likelihood_X = []
        for lp_i, y_i in zip(log_prob_X, y):  # Save probabilities of the true class
            ll = np.zeros(self.n_classes)
            ll[y_i] = lp_i[y_i]
            log_likelihood_X.append(ll)
        log_likelihood_X = np.asarray(log_likelihood_X)

        self.log_likelihoods += np.sum(log_likelihood_X, axis=0)

        # Compute Gradients on updated parameters
        gradient_X = dict()
        dot_param_X = np.dot(X, self.linear_model.coef_.T)

        if self.n_classes == 2:  # logit model
            sigmoid = 1 / (1 + np.exp(-dot_param_X))
            gradient_X[0] = (y.reshape(-1, 1) - sigmoid) * X
            gradient_X[1] = gradient_X[0]  # there's just a single gradient in the binary case
        else:  # multinomial logit
            dot_param_X -= np.max(dot_param_X)  # we subtract a large constant to avoid numpy overflow
            sum_exp = np.sum(np.exp(dot_param_X), axis=1)

            for c in range(self.n_classes):
                kron_delta = y == c
                softmax = np.exp(dot_param_X[:, c]) / sum_exp
                gradient_X[c] = X * (kron_delta - softmax).reshape(-1, 1)

        return log_likelihood_X, gradient_X

    def _add_and_replace_candidates(self, X, log_likelihood_X, gradient_X):
        """ Add new and replace candidate splits in the node statistics
        Identify the split candidates with highest gain (i.e. smallest AIC) in the given data sample.
        Replace partition of old candidates, where the current gain of a new candidate exceeds the old gain.
        Add new candidates if the max size of saved statistics has not been reached yet.

        Parameters
        ----------
        X - (np.array) vector/matrix of observations
        log_likelihood_X - (np.array) log likelihoods regarding each observation in X
        gradient_X - (dict) gradients regarding the parameters of each class

        Returns
        -------
        current_cand_aic - (dict) aic of all currently saved split candidates
        """
        # Update statistics and compute AIC of current candidates
        current_cand_aic = dict()

        for cand in self.log_likelihoods_left.keys():
            if cand[0] in self.cat_features:  # select left child observations
                idx_left = X[:, cand[0]] == cand[1]
            else:
                idx_left = X[:, cand[0]] <= cand[1]

            # Update statistics for potential left and right children
            self.counts_left[cand] += np.count_nonzero(idx_left)
            self.log_likelihoods_left[cand] += np.sum(log_likelihood_X[idx_left], axis=0)
            self.gradients_left[cand] += np.asarray(
                [np.sum(gradient_X[c][idx_left], axis=0) for c in range(self.n_classes)])

            self.counts_right[cand] += np.count_nonzero(~idx_left)
            self.log_likelihoods_right[cand] += np.sum(log_likelihood_X[~idx_left], axis=0)
            self.gradients_right[cand] += np.asarray(
                [np.sum(gradient_X[c][~idx_left], axis=0) for c in range(self.n_classes)])

            # Compute AIC
            current_cand_aic[cand] = self._aic(cand=cand)

        # Allow replacement of x% of the highest ranking candidates (i.e. candidates with worst AIC)
        replaceable_cand_aic = dict(sorted(current_cand_aic.items(), key=lambda item: item[1])[-math.ceil(
            self.n_saved_candidates * self.p_replaceable_candidates):])

        old_candidates = set(self.log_likelihoods_left.keys())

        for ftr in range(self.n_features):
            if ftr in self.cat_features:
                uniques = np.unique(X[:, ftr])
            else:
                uniques = np.unique(np.around(X[:, ftr], decimals=2))  # Round to two decimals and select uniques

            for val in uniques:
                if (ftr, val) not in old_candidates:  # only replace by candidates that are not already saved
                    if ftr in self.cat_features:
                        idx_left = X[:, ftr] == val
                    else:
                        idx_left = X[:, ftr] <= val
                    aic = self._aic(cand=(ftr, val), idx_left=idx_left, log_likelihood_X=log_likelihood_X, gradient_X=gradient_X)

                    # Find first existing candidate with larger aic
                    replace_cand = next((cand for cand in replaceable_cand_aic.keys()
                                          if replaceable_cand_aic[cand] > aic), None)

                    if replace_cand and len(self.log_likelihoods_left.keys()) == self.n_saved_candidates:
                        del self.counts_left[replace_cand]  # Delete outdated candidate
                        del self.log_likelihoods_left[replace_cand]
                        del self.gradients_left[replace_cand]
                        del self.counts_right[replace_cand]
                        del self.log_likelihoods_right[replace_cand]
                        del self.gradients_right[replace_cand]
                        del current_cand_aic[replace_cand]
                        del replaceable_cand_aic[replace_cand]

                    # Add new candidate as long as max size is not reached
                    if len(self.log_likelihoods_left.keys()) < self.n_saved_candidates:
                        self.counts_left[(ftr, val)] = np.count_nonzero(idx_left)
                        self.log_likelihoods_left[(ftr, val)] = np.sum(log_likelihood_X[idx_left], axis=0)
                        self.gradients_left[(ftr, val)] = np.asarray(
                            [np.sum(gradient_X[c][idx_left], axis=0) for c in range(self.n_classes)])

                        self.counts_right[(ftr, val)] = np.count_nonzero(~idx_left)
                        self.log_likelihoods_right[(ftr, val)] = np.sum(log_likelihood_X[~idx_left], axis=0)
                        self.gradients_right[(ftr, val)] = np.asarray(
                            [np.sum(gradient_X[c][~idx_left], axis=0) for c in range(self.n_classes)])

                        current_cand_aic[(ftr, val)] = aic
                        replaceable_cand_aic[(ftr, val)] = aic

        return current_cand_aic

    def _check_for_split(self, candidate_aic):
        """ Check if we need to split the node
        Identify the split candidate with top gain and check whether there is enough evidence to split.

        Parameters
        ----------
        candidate_aic - (dict) aic of all currently saved split candidates

        Returns
        -------
        do_split - (bool) indicator whether to do a split or not
        top_split - (tuple) top feature/value pair used for splitting
        """
        # Get best split candidate with minimal AIC
        cand = min(candidate_aic, key=candidate_aic.get)
        aic_cand = candidate_aic[cand]

        # AIC for a leaf node
        k = self.n_features * self.n_classes if self.n_classes > 2 else self.n_features  # no. of free parameters
        aic_leaf = 2 * k - 2 * np.max(self.log_likelihoods)

        # Perform a statistical test based on the corrected Akaike Information Criterion
        if self.is_leaf:
            # Test at leaf node
            if aic_cand < aic_leaf and math.exp((aic_cand - aic_leaf) / 2) <= self.epsilon:
                return True, cand  # make split
            else:
                return False, cand  # make no split
        else:
            # AIC of the subtree
            log_like_subtree, leaf_count = Node._sum_leaf_likelihoods(self)
            k = leaf_count * self.n_features * self.n_classes if self.n_classes > 2 else leaf_count * self.n_features
            aic_subtree = 2 * k - 2 * np.max(log_like_subtree)

            # Test at inner node
            if aic_leaf < aic_subtree and math.exp((aic_leaf - aic_subtree) / 2) <= self.epsilon:
                return False, None  # Prune and make leaf
            elif aic_cand < aic_subtree and math.exp((aic_cand - aic_subtree) / 2) <= self.epsilon:
                return True, cand  # Prune and replace with another candidate
            else:
                return True, self.split  # Retain current split

    def _aic(self, cand, idx_left=None, log_likelihood_X=None, gradient_X=None):
        """ Compute the Akaike Information Criterion of a given split candidate

        Parameters
        ----------
        cand - (tuple) feature value pair for which we compute the AIC
        idx_left - (np.array) bool array indicating all current observations that fall to the left child
        log_likelihood_X - (np.array) log likelihoods regarding each observation in X
        gradient_X - (np.array) gradients of current observations for each class

        Returns
        -------
        aic - (float) AIC score
        """
        log_like = np.zeros_like(self.log_likelihoods)

        if cand in self.counts_left:  # If candidate is already stored
            count_left = self.counts_left[cand]
            count_right = self.counts_right[cand]

            # As we replace split candidates over time, the candidate stats might not have the same number
            # of observations as the parent node. To compare likelihoods, we scale the candidate likelihood to the
            # same range as the parent node.
            relative_frac = np.divide(self.log_likelihoods_left[cand],
                                      self.log_likelihoods_left[cand] + self.log_likelihoods_right[cand],
                                      out=np.zeros_like(self.log_likelihoods),
                                      where=(self.log_likelihoods_left[cand] + self.log_likelihoods_right[cand]) != 0)
            likelihood_left = self.log_likelihoods * relative_frac

            relative_frac = np.ones_like(self.log_likelihoods) - relative_frac
            likelihood_right = self.log_likelihoods * relative_frac

            gradient_left = self.gradients_left[cand]
            gradient_right = self.gradients_right[cand]

        else:  # If candidate is not already stored, we approximate the likelihood with just the current sample
            count_left = np.count_nonzero(idx_left)
            count_right = np.count_nonzero(~idx_left)

            # As we replace split candidates over time, the candidate stats might not have the same number
            # of observations as the parent node. To compare likelihoods, we scale the candidate likelihood to the
            # same range as the parent node.
            relative_frac = np.divide(np.sum(log_likelihood_X[idx_left], axis=0),
                                      np.sum(log_likelihood_X, axis=0),
                                      out=np.zeros_like(self.log_likelihoods),
                                      where=np.sum(log_likelihood_X, axis=0) != 0)
            likelihood_left = self.log_likelihoods * relative_frac

            relative_frac = np.ones_like(self.log_likelihoods) - relative_frac
            likelihood_right = self.log_likelihoods * relative_frac

            gradient_left = np.asarray([np.sum(gradient_X[c][idx_left], axis=0) for c in range(self.n_classes)])
            gradient_right = np.asarray([np.sum(gradient_X[c][~idx_left], axis=0) for c in range(self.n_classes)])

        for count, likelihood, gradient in zip([count_left, count_right],
                                               [likelihood_left, likelihood_right],
                                               [gradient_left, gradient_right]):
            if count > 0:
                xmax = np.max(np.abs(gradient), axis=1)  # we normalize the gradients to avoid numpy overflow in linalg.norm
                xmax[xmax == 0] = 1  # replace xmax 0 by 1 to avoid division by zero
                norm = np.linalg.norm(gradient / xmax.reshape(-1, 1), axis=1) * xmax
                log_like += likelihood + (self.learning_rate / count) * norm ** 2

        k = self.n_features * self.n_classes if self.n_classes > 2 else self.n_features  # no. of free param

        return 2 * k - 2 * np.max(log_like)

    @staticmethod
    def _sum_leaf_likelihoods(node, likelihoods=0, leaf_count=0):
        """ Sum up the likelihoods at the leaves of a subtree

        Parameters
        ----------
        node - (Node) current node in the DMT
        likelihoods - (int) sum of likelihoods
        leaf_count - (int) count of leaves

        Returns
        -------
        likelihoods - (int) updated sum of likelihoods
        leaf_count - (int) updated count of leaves
        """
        if node.is_leaf:
            return likelihoods + node.log_likelihoods, leaf_count + 1
        else:
            likelihoods, leaf_count = Node._sum_leaf_likelihoods(node.children[0], likelihoods, leaf_count)
            likelihoods, leaf_count = Node._sum_leaf_likelihoods(node.children[1], likelihoods, leaf_count)
            return likelihoods, leaf_count

    def predict_observation(self, x, get_prob=False):
        """ Predict one observation (recurrent function)
        Pass observation down the tree until a leaf is reached. Make prediction at leaf.

        Parameters
        ----------
        x - (np.array) observation vector
        get_prob - (bool) indicator whether to return class probabilities

        Returns
        -------
        y_pred/y_prob - (np.array) predicted class label/probability of the given observation
        """
        if self.is_leaf:
            x = x.reshape(1, -1)
            if get_prob:
                return self.linear_model.predict_proba(x)
            else:
                return self.linear_model.predict(x)
        else:
            if self.split[0] in self.cat_features:
                if x[self.split[0]] == self.split[1]:  # advance to children
                    return self.children[0].predict_observation(x)
                else:
                    return self.children[1].predict_observation(x)
            else:
                if x[self.split[0]] <= self.split[1]:  # advance to children
                    return self.children[0].predict_observation(x)
                else:
                    return self.children[1].predict_observation(x)
