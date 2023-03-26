import numpy as np
from utils import Visualizer


def polar_kernel_matrix(X1, X2):
    """
    Matrix output version of polar_kernel. Sklearn expects
    custom kernel functions to be in this format for performance purposes
    :param X1: A numpy matrix of size n_samples x features
    :param X2: A numpy matrix of the same dimensions as X1
    :return: A n_sample x n_sample matrix where each element is K(Xi, Xj)
    """
    squared_norm1 = np.linalg.norm(X1, axis=1)**2
    squared_norm2 = np.linalg.norm(X2, axis=1)**2
    return np.inner(X1, X2) + np.outer(squared_norm1, squared_norm2)


def polar_kernel(X1, X2):
    """
    Returns K(X1, X2) where K is the kernel represenation of <phi(X1), phi(X2)>
    and phi is the polar embedding.
    :param X1: a numpy row vector, representing a single sample
    :param X2: a numpy row vector, representing a single sample
    :return: a scalar value representing K(X1, X2)
    """
    # TODO
    pass


def polar_embedding(X):
    """
    Returns X in higher dimensional feature space, in otherwords, adds a feature to each sample in X.
    The added feature should be the right most column.
    :param X: a n_sample x n_features numpy matrix of samples. In part 1&2, X has 2 original features.
    :return: a n_sample x n_features + 1 matrix of samples with the additional feature appended on the right
    """
    # TODO
    pass


class SolutionVisualization(Visualizer):
    def __init__(self, svm, kernel=None, embedding_function=None):
        super().__init__()
        self.svm = svm
        self.kernel = kernel
        self.embedding_function = embedding_function

    def generate_new_feature_values(
            self, gridpoints_x1, gridpoints_x2):
        """
        Generates x_1^2 + x_2^2 values which lie on H(x) = 0 for each gridpoint (x_1, x_2)
        :param gridpoints_x1: A n_axis_len x n_axis_len (see utils.py for defition) numpy matrix of values representing x_1 values
        :param gridpoints_x2: A n_axis_len x n_axis_len numpy matrix of values representing x_2 values
        :return: A n_axis_len x n_axis_len numpy matrix of values representing x_1^2 + x_2^2 values for each corresponding gridpoint
        """

        # the weight vector for H, shape: 1 x n_features
        coefficients = self.svm.coef_.flatten()
        # the scalar bias term
        intercept = self.svm.intercept_.flatten()
        
        # TODO: Calculate x2^2 + x1^2 points which lie on H(x) using the equation given in the handout
        new_feature_points = None

        return new_feature_points

    def _H(self, X, x_input, dual_coefficients, intercept):
        """
        Calculates H using formula (9) in the handout
        :param X: a 2d numpy array where each row is a training sample.
        :param x_input: a 1 x n_features numpy array representing a single data point to evaluate H on
        :param dual_coefficients: numpy vector, notated as alpha in the handout
        :param intercept: a scalar representing the bias term
        :return: A scalar, the value of H(x_input)
        """
        # TODO: Compute K(X_i, x_input) for each training data point in X.
        kernel_vals = None

        # TODO: Implement the equation for H
        H = None
        return H

    def generate_H_points(
            self, X, gridpoints_x1, gridpoints_x2):
        """
        Generates an H value for each gridpoint (x_1, x_2)
        :param X: A n_training_samples x n_features matrix of training data
        :param gridpoints_x1: A n_axis_len x n_axis_len numpy matrix of values representing x_1 values
        :param gridpoints_x2: A n_axis_len x n_axis_len numpy matrix of values representing x_2 values
        :return: A n_axis_len x n_axis_len numpy matrix of values representing H values for each corresponding gridpoint
        """
        # Note: the dual coefficients are notated as alpha in the handout and the slides
        # Note: sklearn does not store dual coefficients of non support vectors
        # TODO: Get the dual coefficients and intercept from self.svm
        dual_coefficients = np.zeros(len(X))
        intercept = None

        # TODO: Get the value of H at each gridpoint
        # gridpoints here is a n_gridpoints x 2 matrix where each row is a (x_1, x_2) gridpoint
        gridpoints = np.c_[gridpoints_x1.ravel(), gridpoints_x2.ravel()]

        # Think about what you should be passing in as X to _H. Does H(x) always depend on all training points?
        H = None

        return H

    def visualize_H(
            self,
            X,
            gridpoints_x1,
            gridpoints_x2,
            plot_params={}):
        """
        Generates a plot of H(x)
        :param X: A n_training_sampels x n_features numpy matrix of training samples
        :param gridpoints_x1: A n_axis_len x n_axis_len numpy matrix of values representing x_1 values
        :param gridpoints_x2: A n_axis_len x n_axis_len numpy matrix of values representing x_2 values
        :return: None, a plot should appear when called
        """
        H = self.generate_H_points(X, gridpoints_x1, gridpoints_x2)

        self.visualize_3d(
            gridpoints_x1, gridpoints_x2, H, plot_params=plot_params
        )
