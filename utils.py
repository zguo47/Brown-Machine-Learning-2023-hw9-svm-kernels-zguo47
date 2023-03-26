import numpy as np
import matplotlib.pyplot as plt

# If you're plots are too large,try decreasing this (must be an integer)
PLOT_SIZE_MAGNIFIER = 4


def load_csv_data(file_path):
    """
    Takes in a file path to a csv file and returns a 2d numpy array
    :param file_path: A string indicating the file path
    :return: A 2d numpy array of the csv values
    """
    csv_data = np.loadtxt(file_path, delimiter=',')
    return csv_data


class Visualizer:
    def __init__(self):
        # Dimensions of the plot to be produced.
        # Feel free to change them, but the current ones are recommended
        self.x_min, self.x_max = -1.5, 1.5
        self.y_min, self.y_max = -1.5, 1.5
        self.z_min, self.z_max = 0, 4

        # The number of points along each axis
        self.axis_len = 100

        # The opacity of the surface being generated in 3d plots
        self.surface_alpha = 0.2

    def get_gridpoints(self):
        """
        Returns two 2d numpy arrays of evenly spaced values
        :return: two numpy arrays of dimensions self.axis_len x self.axis_len
        """
        gridpoints_x, gridpoints_y = np.meshgrid(
            np.linspace(self.x_min, self.x_max, num=self.axis_len),
            np.linspace(self.y_min, self.y_max, num=self.axis_len)
        )

        return gridpoints_x, gridpoints_y

    def visualize_3d(
            self,
            gridpoints_x1,
            gridpoints_x2,
            solution_points,
            input_data=None,
            labels=None,
            plot_params={}):
        """
        Generates a 3d plot of a surface
        :param gridpoints_x1: A n_axis_len x n_axis_len numpy matrix of values representing x_1 values
        :param gridpoints_x2: A n_axis_len x n_axis_len numpy matrix of values representing x_2 values
        :param solution_points: A 1d or 2d numpy array, must be reshapable into dimensions self.axis_len x self.axis_len. Represents
        the values of the surface being plotted
        :param input_data: The feature values of data points to plot
        :param labels: The labels of the data to plot
        :param plot_params: A dictionary of values to modify the plots, ex: title, axis labels
        """
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.plot_surface(
            gridpoints_x1, gridpoints_x2, solution_points.reshape(
                (self.axis_len, self.axis_len)), alpha=0.2)

        if input_data is not None and labels is not None:
            ax.scatter(input_data[:, 0], input_data[:, 1],
                       input_data[:, 2], c=labels)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.z_min, self.z_max)

        if 'title' in plot_params:
            ax.set_title(plot_params['title'])

        if 'ylabel' in plot_params:
            ax.set_ylabel(plot_params['ylabel'])

        if 'xlabel' in plot_params:
            ax.set_xlabel(plot_params['xlabel'])
        
        if 'zlabel' in plot_params:
            ax.set_zlabel(plot_params['zlabel'])

        plt.show()

    def visualize_2d(
            self,
            gridpoints_x1,
            gridpoints_x2,
            solution_points,
            input_data,
            labels,
            plot_params={}):
        """
        Generates 2d contour plots
        :param gridpoints_x1: A n_axis_len x n_axis_len numpy matrix of values representing x_1 values
        :param gridpoints_x2: A n_axis_len x n_axis_len numpy matrix of values representing x_2 values
        :param solution_points: A 2d or 3d numpy array, the inner numpy arrays must be reshapable into dimensions self.axis_len x self.axis_len.
        Represents the values of the surface being plotted
        :param input_data: The feature values of data points to plot
        :param labels: The labels of the data to plot
        :param plot_params: A dictionary of values to modify the plots, ex: title, axis labels
        """
        n_plots = solution_points.shape[0]
        fig, _ = plt.subplots(
            1, n_plots, figsize=(
                PLOT_SIZE_MAGNIFIER * n_plots, PLOT_SIZE_MAGNIFIER))

        for i, ax in enumerate(fig.axes):
            cur_sol_pts = solution_points[i]

            ax.contourf(gridpoints_x1, gridpoints_x2, cur_sol_pts.reshape(
                (self.axis_len, self.axis_len)), alpha=0.5, levels=1)
            ax.scatter(input_data[:, 0], input_data[:, 1], c=labels)

            if 'multi_title' in plot_params:
                ax.set_title(plot_params['multi_title'][i])

            if 'ylabel' in plot_params:
                ax.set_ylabel(plot_params['ylabel'])

            if 'xlabel' in plot_params:
                ax.set_xlabel(plot_params['xlabel'])

            ax.label_outer()
        if 'title' in plot_params:
            fig.suptitle(plot_params['title'])
        plt.show()

