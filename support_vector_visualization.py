from utils import Visualizer
import matplotlib.pyplot as plt
import numpy as np

def separate_support_vectors(arr, sv_idx):
    """
    Separates the 2d numpy array of training examples into a 2d np array of support vectors and a 2d array of non support vectors
    :param arr: The 2d numpy array to be separated
    :param sv_idx: The 1d array of indices indicating which rows to remove from arr and place into support_vectors.
    non_support_vectors should be the remaining rows from arr.
    :return: two numpy arrays with the same number of columns as arr
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    support_vectors = []
    non_support_vectors = []
    for index in range(arr.shape[0]):
        if index in sv_idx:
            support_vectors.append(arr[index])
        else:   
            non_support_vectors.append(arr[index])
    support_vectors = np.asarray(support_vectors).reshape(-1, arr.shape[1])
    non_support_vectors = np.asarray(non_support_vectors).reshape(-1, arr.shape[1])

    return support_vectors, non_support_vectors

class SupportVectorVisualization(Visualizer):
    def __init__(self, svm):
        self.svm = svm

    def generate_x2_vals(
            self, x1_axis_points, coefficients, intercept):
        """
        Calculates the values of x_2 that lie on H(x) = 0 corresponding to each inputted value of x_1
        :param x1_axis_points: A numpy vector of x_1 values
        :param coefficients: A numpy vector of coefficients, [w_1, w_2]
        :param intercept: A scalar value representing the bias
        :return: A numpy vector of x_2 values
        """
        # TODO: Calculate the slope of the decision boundary
        slope = coefficients[0]/coefficients[1]

        # TODO: Calculate the x2 values corresponding to the (x1, x2) points which lie on the decision boundary
        x2_vals = -(coefficients[0] * x1_axis_points + intercept)/coefficients[1]

        return x2_vals

    def generate_margin_points(self, x2_vals, coefficients):
        """
        For each point of the decision boundary, calculates the point on the margin
        below the boundary and above the boundary.
        :param x2_vals: A numpy vector of x_2 values
        :param coefficients: A numpy vector of coefficients, [w_1, w_2]
        :return: A numpy vector representing the upper margin points
                 and a numpy vector representing the lower margin points
        """

        # TODO: Get the vertical distance from the hyperplane to the margin using
        # equation (16) from the handout
        slope = coefficients[0]/coefficients[1]
        vertical_distance = np.sqrt(1 + slope**2)/np.linalg.norm(coefficients)

        # TODO: Calculate the margin points.
        upper_margin_vals, lower_margin_vals = x2_vals + vertical_distance, x2_vals - vertical_distance

        return upper_margin_vals, lower_margin_vals

    def visualize_solution(self, X, Y, x1_axis_points, ax):
        """
        Modifies an axes object by plotting the margins, and the data points in R2. Outlines the support vectors
        :param X: A n_training_samples x n_features numpy matrix of training data
        :param Y: A numpy vector of labels for X
        :param x1_axis_points: A numpy vector of x_1 values
        :param ax: an axes object
        """
        # These variables have n_classes rows, but since we only have 1 class
        # We flatten them into vectors
        coefficients = self.svm.coef_.flatten()
        intercept = self.svm.intercept_.flatten()

        # TODO: Get the x2 and margin values
        x2_vals= self.generate_x2_vals(x1_axis_points, coefficients, intercept)
        upper_margin_vals = self.generate_margin_points(x2_vals, coefficients)[0]
        lower_margin_vals = self.generate_margin_points(x2_vals, coefficients)[1]

        # TODO: Seperate the support vector points from the non support vector
        # points
        support_vectors, non_support_vectors = separate_support_vectors(X, self.svm.support_)
        support_vector_labels, non_support_vector_labels = separate_support_vectors(Y, self.svm.support_)

        # -------- DO NOT MODIFY ANYTHING BELOW THIS LINE --------
        ax.plot(x1_axis_points, x2_vals) # plots (x1, x2)
        ax.plot(x1_axis_points, upper_margin_vals, "k--") # plots (x1, upper margin)
        ax.plot(x1_axis_points, lower_margin_vals, "k--") # plots (x1, lower margin)
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                   s=100, c=support_vector_labels, edgecolors='red')
        ax.scatter(
            non_support_vectors[:, 0], non_support_vectors[:, 1], c=non_support_vector_labels)

        ax.set_ylabel(r"$x_2$")
        ax.set_xlabel(r"$x_1$")
        ax.label_outer()
