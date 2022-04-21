import numpy as np
from sklearn.preprocessing import StandardScaler
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.neural_networks import PerceptronMask
import copy
from scipy import stats
import warnings


class FIMTDDClassifier:
    """ FIMT-DD Classifier.

    A FIMT-DD classification model based on the description and pseudo code by

    Ikonomovska, E., Gama, J. & Džeroski, S. Learning model trees from evolving data streams.
    Data Min Knowl Disc 23, 128–168 (2011). https://doi.org/10.1007/s10618-010-0201-y

    """
    def __init__(self, attr, classes, delta=0.01, n_min=1, tau=0.05, lr=0.01):
        # Set up root node
        self.classes = classes
        self.root = FIMTDD_Node(delta,  # probability for Hoeffding bound
                                n_min,  # no. of observations required per node before attempting to split
                                tau,    # tie-break threshold for split decisions
                                attr,   # all attributes/features to be considered
                                lr)     # learning rate of linear models

    def partial_fit(self, x, y):
        """ Fit FIMTDD to current batch of observations

        Parameters
        ----------
        x observations (n_observations x n_features
        y targets (n_observations)
        """
        # Starting from the root, traverse observations to their corresponding leaf nodes
        self.root, _, _ = self._traverse_tree(x, y, self.root, errors=[])

    def predict(self, x):
        """ Predict given batch of obseravtions

        Parameters
        ----------
        x - observations (n_observations x n_features)

        Returns
        -------
        y_pred (n_observations)
        """
        y_pred = []
        for x_i in x:
            y_pred.append(self._predict_instance(x_i, node=self.root))
        return np.asarray(y_pred).flatten()

    def _predict_instance(self, x_i, node):
        """ Predict a given instance

        Parameters
        ----------
        x_i - data instance (n_features)
        node - current node

        Returns
        -------
        y_i_pred
        """
        if node.is_leaf:
            return node.linear_model.predict(x_i.reshape(1, -1))
        else:
            if x_i[node.split[0]] <= node.split[1] and node.children[0] is not None:  # traverse to left child
                return self._predict_instance(x_i, node.children[0])
            elif x_i[node.split[0]] > node.split[1] and node.children[1] is not None:  # traverse to right child
                return self._predict_instance(x_i, node.children[1])
            else:  # if corresponding child node does not exist (due to change adaptation), predict at current node
                # x = (x - self.scaler.mean_) / (3 * self.scaler.scale_)  # Scale data  Note: not required for experiments, since data is already scaled!
                return node.linear_model.predict(x_i.reshape(1, -1))

    def n_nodes(self):
        """ Return the number of nodes (total), leaves and the depth

        Returns
        -------
        n_total, n_leaf, depth
        """
        return self._add_up_node_count(n_total=0, n_leaf=0, depth=0, node=self.root)

    def _add_up_node_count(self, n_total, n_leaf, depth, node):
        """ Increment node counts

        Parameters
        ----------
        n_total - current total no. of nodes
        n_leaf - current no. of leaves
        depth - current depth
        node - current node

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

    def _traverse_tree(self, x, y, node, errors):
        """ Traverse the given observations to a corresponding leaf node
        & update leaf statistics & check for change along the way

        Parameters
        ----------
        x - observations (n_observations, n_features)
        y - target (n_observations)
        node - current node
        errors - array of 0-1 loss measures (error)

        Returns
        -------
        node (updated), is_bad,  errors
        """
        if node.is_leaf:
            # Update linear model and statistics
            split = node.update(x, y, classes=self.classes)

            # Append errors (required for change detection)
            # x = (x - self.scaler.mean_) / (3 * self.scaler.scale_)  # Scale data  Note: not required for experiments, since data is already scaled!
            y_pred = node.linear_model.predict(x)
            errors.extend((y_pred == y) * 1)

            if split:  # split leaf node
                # Create children
                children = []
                for i in range(2):
                    child = FIMTDD_Node(delta=node.delta, n_min=node.n_min, tau=node.tau, attr=node.attr, lr=node.lr)
                    child.linear_model = copy.deepcopy(node.linear_model)  # populate linear model
                    children.append(child)

                # Reset node (and make inner node)
                node = FIMTDD_Node(delta=node.delta, n_min=node.n_min, tau=node.tau, attr=node.attr, lr=node.lr)
                node.linear_model = copy.deepcopy(children[0].linear_model)  # save linear model to avoid performance drop if node turns leaf again
                node.is_leaf = False
                node.split = split
                node.children = children
        else:
            left_idx = x[:, node.split[0]] <= node.split[1]  # split samples acc to split attribute
            right_idx = x[:, node.split[0]] > node.split[1]

            # Traverse samples to corresponding child nodes
            left_bad = False
            right_bad = False
            if any(left_idx):
                node.children[0], left_bad, errors = self._traverse_tree(x[left_idx], y[left_idx], node.children[0], errors)  # left
            if any(right_idx):
                node.children[1], right_bad, errors = self._traverse_tree(x[right_idx], y[right_idx], node.children[1], errors)  # right

            # Drop old (bad/outdated) nodes and branches
            if left_bad and right_bad:  # make leaf, if both children are outdated
                node.is_leaf = True
                node.children = []
                node.split = None

        # Check for change (concept drift)
        is_bad = False
        for err in errors:  # update Page Hinkley
            node.change_detector.add_element(err)
            if node.change_detector.detected_change():
                is_bad = True

        return node, is_bad, errors


class FIMTDD_Node:
    def __init__(self, delta, n_min, tau, attr, lr=0.01):
        self.delta = delta  # probability for Hoeffding bound
        self.n_min = n_min  # no. of observations required per node before attempting to split
        self.tau = tau      # tie-break threshold for split decisions
        self.attr = attr    # all attributes/features to be considered
        self.lr = lr        # learning rate for linear model

        self.n = 0              # observation count (since last split attempt)
        self.is_leaf = True     # indicator of leaf node
        self.children = []      # ids of child nodes
        self.split = None       # feature/value tuple of split attribute
        self.bst_roots = dict() # root nodes per attribute of a (extended) Binary Search Tree

        self.scaler = StandardScaler()                  # scaler for online standardization of samples
        self.linear_model = PerceptronMask(eta0=lr)     # initialize Perceptron model with learning rate (weights are initialied acc. to sklearn procedure)

        self.change_detector = PageHinkley()  # set up change detector (default parameters due to Ikonomovska)

        self.curr_top_sdr = 0         # current top sdr
        self.curr_top_ratio = 1       # current top sdr ratio
        self.top_sdr_ratios = dict()  # sdr scores of top split candidates in consecutive time steps

    def update(self, x, y, classes):
        """ Update Leaf Node Statistics and fit linear model

        Parameters
        ----------
        x - observations (n_observations, n_features)
        y - target (n_observations)
        classes - unique class labels

        Returns
        -------
        candidate split or FALSE
        """
        self.n += len(y)  # increment observation count

        # Scale data  Note: note required for experiments, since data is already scaled!
        #self.scaler.partial_fit(x)
        #x = (x - self.scaler.mean_) / (3 * self.scaler.scale_)

        # Fit linear model
        self.linear_model.partial_fit(x, y, classes=classes)

        # Update statistics of BST of each attribute
        for x_i, y_i in zip(x, y):
            for attr in self.attr:
                if attr not in self.bst_roots:  # add new attribute
                    self.bst_roots[attr] = BSTNode(x_i[attr])
                self.bst_roots[attr].insert(x_i[attr], y_i)  # insert value

        if self.n >= self.n_min:  # after every n_min observations -> attempt to split
            self.n = 0
            best = None
            best_val = None
            sec_best = None
            sec_best_val = None
            best_sdr = 0
            sec_best_sdr = 0

            # Find best & second best split candidate attributes acc. to standard deviation reduction
            # Note that FIMT considers only one split candidate per feature
            for attr in self.attr:
                # Extract number of random variables in Hoeffding Test -> required to cancel bad split points in the BST
                # We extract the largest number as an approximation
                attr_n = 1
                for attrs, vals in self.top_sdr_ratios.items():
                    if len(vals['ratios']) > attr_n:
                        attr_n = len(vals['ratios'])

                _, attr_best_val, attr_best_sdr, _, _, _, _ = self.bst_roots[attr].find_best_split(
                    prev_top_sdr=self.curr_top_sdr, prev_top_ratio=self.curr_top_ratio, delta=self.delta, attr_n=attr_n)

                if attr_best_sdr > best_sdr:
                    sec_best = best
                    sec_best_val = best_val
                    sec_best_sdr = best_sdr
                    best = attr  # replace best split candidate
                    best_val = attr_best_val
                    best_sdr = attr_best_sdr
                elif attr_best_sdr > sec_best_sdr:
                    sec_best = attr  # replace second best candidate
                    sec_best_val = attr_best_val
                    sec_best_sdr = attr_best_sdr

            # Update current top SDR value and ratio (required for detection of 'bad' BST nodes)
            if best_sdr > 0 and sec_best_sdr > 0:  # only update if we actually found a good attr in terms of the sdr
                self.curr_top_sdr = best_sdr
                self.curr_top_ratio = sec_best_sdr / best_sdr

                # Save sdr ratio and values of best split candidate at this time step
                if (best, sec_best) in self.top_sdr_ratios:
                    self.top_sdr_ratios[(best, sec_best)]['best_val'].append(best_val)
                    self.top_sdr_ratios[(best, sec_best)]['sec_best_val'].append(sec_best_val)
                    self.top_sdr_ratios[(best, sec_best)]['ratios'].append(sec_best_sdr / best_sdr)
                else:
                    self.top_sdr_ratios[(best, sec_best)] = dict()
                    self.top_sdr_ratios[(best, sec_best)]['best_val'] = [best_val]
                    self.top_sdr_ratios[(best, sec_best)]['sec_best_val'] = [sec_best_val]
                    self.top_sdr_ratios[(best, sec_best)]['ratios'] = [sec_best_sdr / best_sdr]

            # Apply Hoeffding Bound to candidate with most measurements to check if split ratios suggest split
            top_attr = None
            top_attr_val = None
            top_mean_ratio = 100
            top_n = 1
            for attrs, vals in self.top_sdr_ratios.items():
                if len(vals['ratios']) > top_n:
                    top_n = len(vals['ratios'])
                    top_attr = attrs[0]
                    top_attr_val = stats.mode(vals['best_val'])[0][0]  # use most frequent value as value to split on
                    top_mean_ratio = np.mean(vals['ratios'])

            # Compute Hoeffding Bound
            eps = np.sqrt(np.log(1 / self.delta) / (2 * top_n))

            if top_mean_ratio + eps < 1:
                return tuple((top_attr, top_attr_val))  # return candidate attribute and value to split on

            if eps < self.tau and top_attr is not None:  # tie break check
                print('Tie Break! Enforcing Split...')
                return tuple((top_attr, top_attr_val))  # return any attribute and value, as all are equally good

        return False


class BSTNode:
    def __init__(self, val):
        self.val = val              # attribute value represented by node
        self.left_child = None      # left child (smaller value)
        self.right_child = None     # right child (greater value)

        self.n_left = 0             # sum of instances falling to the left of the node
        self.sum_y_left = 0         # sum of y (target variable) falling to the left
        self.sq_sum_y_left = 0      # sum of squared y falling to the left
        self.n_right = 0            # sum of instances falling to the left of the node
        self.sum_y_right = 0        # sum of y (target variable) falling to the left
        self.sq_sum_y_right = 0     # sum of squared y falling to the left

    def insert(self, x_attr_i, y_i):
        """ Insert new value

        Parameters
        ----------
        x_attr_i attribute value of x instance (float)
        y_i corresponding label
        """
        # Update statistics
        if x_attr_i <= self.val:  # update or create left child
            self.n_left += 1
            self.sum_y_left += y_i
            self.sq_sum_y_left += y_i ** 2

            if x_attr_i < self.val:
                if not self.left_child:
                    self.left_child = BSTNode(x_attr_i)
                self.left_child.insert(x_attr_i, y_i)
        elif x_attr_i > self.val:  # update or create right child
            self.n_right += 1
            self.sum_y_right += y_i
            self.sq_sum_y_right += y_i ** 2

            if not self.right_child:
                self.right_child = BSTNode(x_attr_i)
            self.right_child.insert(x_attr_i, y_i)

    def find_best_split(self, prev_top_sdr, prev_top_ratio, delta, attr_n, best_val=None, best_sdr=0, sum_total_left=0,
                        sum_total_right=0, sq_sum_total_left=0, sq_sum_total_right=0, right_total=0, total=0):
        """ Find best split value acc. to SDR
        (According to PseudoCode by Ikonomovska et al.)

        Parameters
        ----------
        prev_top_sdr - previous top sdr score (required for bad-split check)
        prev_top_ratio - previous top sdr ratio (required for bad-split check)
        delta - probability for computation of Hoeffding bound
        attr_n - sample count to identify bad split counts in BST
        best_val - current best value
        best_sdr - current best sdr
        sum_total_left - total sum of y values in left branch
        sum_total_right - total sum of y value in right branch
        sq_sum_total_left - total squared sum of y values in left branch
        sq_sum_total_right - total squared sum of y values in right branch
        right_total - total no. of instances in right branch
        total - total no. of instances observed

        Returns
        -------
        is_bad (bool), best_val, best_sdr, sum_total_left, sum_total_right, sq_sum_total_left, sq_sum_total_right
        """
        # Initialize sums at root node
        if sum_total_right == 0 and sq_sum_total_right == 0:
            sum_total_right = self.sum_y_left + self.sum_y_right
            sq_sum_total_right = self.sq_sum_y_left + self.sq_sum_y_right
            right_total = self.n_left + self.n_right
            total = right_total

        # Indicator whether children are bad splits
        left_is_bad = False if self.left_child else True  # if a child does not exist, set to true to avoid blocking pruning
        right_is_bad = False if self.right_child else True

        if self.left_child:  # check left child branch first
            left_is_bad, best_val, best_sdr, sum_total_left, sum_total_right, sq_sum_total_left, sq_sum_total_right = self.left_child.find_best_split(
                prev_top_sdr=prev_top_sdr,
                prev_top_ratio=prev_top_ratio,
                delta=delta,
                attr_n=attr_n,
                best_val=best_val,
                best_sdr=best_sdr,
                sum_total_left=sum_total_left,
                sum_total_right=sum_total_right,
                sq_sum_total_left=sq_sum_total_left,
                sq_sum_total_right=sq_sum_total_right,
                right_total=right_total,
                total=total
            )

        # Update sums in order to compute SDR
        sum_total_left += self.sum_y_left
        sum_total_right -= self.sum_y_left
        sq_sum_total_left += self.sq_sum_y_left
        sq_sum_total_right -= self.sq_sum_y_left
        right_total -= self.n_left

        node_sdr = self._sdr(sum_total_left, sum_total_right, sq_sum_total_left, sq_sum_total_right, right_total, total)

        # Check if top splits can be replaced
        if node_sdr > best_sdr:
            best_val = self.val
            best_sdr = node_sdr

        if self.right_child:  # check right child branch
            right_is_bad, best_val, best_sdr, sum_total_left, sum_total_right, sq_sum_total_left, sq_sum_total_right = self.right_child.find_best_split(
                prev_top_sdr=prev_top_sdr,
                prev_top_ratio=prev_top_ratio,
                delta=delta,
                attr_n=attr_n,
                best_val=best_val,
                best_sdr=best_sdr,
                sum_total_left=sum_total_left,
                sum_total_right=sum_total_right,
                sq_sum_total_left=sq_sum_total_left,
                sq_sum_total_right=sq_sum_total_right,
                right_total=right_total,
                total=total)

        # Update sums to propagate to parent
        sum_total_left -= self.sum_y_left
        sum_total_right += self.sum_y_left
        sq_sum_total_left -= self.sq_sum_y_left
        sq_sum_total_right += self.sq_sum_y_left
        right_total += self.n_left

        # Check if current split is bad (non-promising regarding Hoeffding Bound)
        eps = np.sqrt(np.log(1 / delta) / (2 * attr_n))
        if prev_top_sdr > 0 and node_sdr / prev_top_sdr < prev_top_ratio - 2 * eps:
            is_bad = True
        else:
            is_bad = False

        if is_bad and left_is_bad and right_is_bad:  # if current node and all children are bad splits -> prune
            self.left_child = None
            self.right_child = None
            return True, best_val, best_sdr, sum_total_left, sum_total_right, sq_sum_total_left, sq_sum_total_right
        else:
            return False, best_val, best_sdr, sum_total_left, sum_total_right, sq_sum_total_left, sq_sum_total_right

    @staticmethod
    def _sdr(sum_total_left, sum_total_right, sq_sum_total_left, sq_sum_total_right, right_total, total):
        """ Compute Standard Deviation Reduction

        Returns
        -------
        sdr
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                sd = np.sqrt(((sq_sum_total_left + sq_sum_total_right) - (sum_total_left + sum_total_right) ** 2 / total) / total)

                if total - right_total > 0:
                    sd -= (total - right_total) / total * np.sqrt(
                            (sq_sum_total_left - sum_total_left ** 2 / (total - right_total)) / (total - right_total))

                if right_total > 0:
                    sd -= right_total / total * np.sqrt(
                            (sq_sum_total_right - sum_total_right ** 2 / right_total) / right_total)

                return sd
            except Warning:
                # Depending on the target it can happen that the sum of squares is smaller than the squared sum,
                # in which case the value we apply the square root to is negative.
                # To avoid a warning, we return a small positive constant instead
                return 10e-7
