import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Function to plot bar charts in a grid
def plot_grid_bar(data, columns, by=None):
    """
    Grid-wise plots the barplots for specified columns of a dataset. With hue split.
    Args:
        data (pandas DataFrame): Dataframe
        columns (list): Column list to be plotted
        by: binomial variable we want to hue on
    """

    num_plots = len(columns)
    rows = (num_plots + 1) // 2

    # Create subplots grid
    fig, ax = plt.subplots(nrows=rows, ncols=2, figsize=(12, 4 * rows))

    # Iterate over the columns and plot their distributions
    for i, column in enumerate(columns):
        row = i // 2
        col = i % 2

        # Plot barplot in respective subplot
        sns.countplot(data=data, x=column, ax=ax[row, col], hue=by)
        ax[row, col].set_ylabel('Count')
        ax[row, col].set_xlabel(column)
        ax[row, col].set_title(column)

        # Annotate value counts on each bar
        for p in ax[row, col].patches:
            height = p.get_height()
            ax[row, col].annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                                  ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Delete the last subplot if the number of plots is odd
    if num_plots % 2 != 0:
        fig.delaxes(ax[-1,1])

    # Layout spacing
    fig.tight_layout()

    # Show plot
    plt.show()


# Function to plot distributions of columns with median and mean in a tiled grid
def plot_grid_displots(data, columns, units_list):
    """
    Grid-wise plots the distribution of values for specified columns of a dataset.
    Args:
        data (pandas DataFrame): Dataframe
        columns (list): Column list to be plot
        units_list (list): Measurement units for each column
    """

    num_plots = len(columns)
    rows = (num_plots + 1) // 2

    # Create subplots grid
    fig, ax = plt.subplots(nrows=rows, ncols=2, figsize=(12, 4 * rows))

    # Iterate over the columns and plot their distributions
    for i, column in enumerate(columns):
        row = i // 2
        col = i % 2

        # Plot displot in subplot
        sns.histplot(data=data, x=column, ax=ax[row, col], kde=True)
        ax[row, col].set_title(column)
        ax[row, col].set_xlabel(f'{column} ({units_list[i]})')
        ax[row, col].set_ylabel(f'Count')

        # Add vertical lines for median and mean
        median = data[column].median()
        mean = data[column].mean()
        ax[row, col].axvline(median, color='r', linestyle='dotted',
                             label=f'Median = {data[column].median():.2f} {units_list[i]}')
        ax[row, col].axvline(mean, color='g', linestyle='dotted',
                             label=f'Mean = {data[column].mean():.2f} {units_list[i]}')

        ax[row, col].legend()

    # Delete the last subplot if the number of plots is odd
    if num_plots % 2 != 0:
        fig.delaxes(ax[-1,1])

    # Layout spacing
    fig.tight_layout()

    # Show plot
    plt.show()


def plot_grid_violin(data, binom, continuous_vars, binom_vars, title_fontsize=19):
    """
    Grid-wise plots the violin plots for specified continuous variables against discrete
    binomial variable and discrete binomial variables as hue in a dataset.
    Args:
        data (pandas DataFrame): Dataframe
        binom (string): Binomial variable to split violins by
        continuous_vars (list): Continuous variable list to be plotted
        binom_vars (list): Discrete binomial variable list to be plotted
        title_fontsize (int): Font size for subplot titles, default is 20
    """
    num_continuous_vars = len(continuous_vars)
    num_discrete_vars = len(binom_vars)

    # Create subplots grid
    fig, ax = plt.subplots(num_continuous_vars,
                           num_discrete_vars,
                           figsize=(6 * num_discrete_vars, 4 * num_continuous_vars))

    # Iterate over continuous variables
    for i, continuous_var in enumerate(continuous_vars):
        # Iterate over discrete variables
        for j, discrete_var in enumerate(binom_vars):
            # Plot violin plot in respective subplot
            sns.violinplot(data=data, x=binom, y=continuous_var, hue=discrete_var, split=True, ax=ax[i, j])
            ax[i, j].set_xlabel(binom)
            ax[i, j].set_ylabel(continuous_var)
            ax[i, j].legend(title=discrete_var)

            # Add subplot title
            title = f'{continuous_var} vs {binom} by {discrete_var}'
            ax[i, j].set_title(title, fontsize=title_fontsize)

    # Layout spacing
    fig.tight_layout()

    # Show plot
    plt.show()


# Function that plots a grid of box plots
def plot_box_grid(data, cols, figsize=(15, 15)):
    """
    Grid-wise plots the distribution of values for specified columns of a dataset.
    Args:
        data (pandas DataFrame): Dataframe
        cols (list): Columns to be plotted
        figsize (tuple): Size of the figure (optional, default=(15, 15)
    """

    # Calculate the number of rows and columns based on the number of columns in columns_list
    num_columns = len(cols)
    num_rows = math.ceil(num_columns / 4)

    # Create the subplot grid
    fig, ax = plt.subplots(num_rows, 4, figsize=figsize)

    # Flatten the ax array if the grid is not perfect (e.g., 2x4 instead of 4x4)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = np.array([ax])

    # Loop through the columns and draw the box plots
    for i, column in enumerate(cols):
        data.boxplot(column, ax=ax[i])
        ax[i].set_title(column)

    # Remove any empty subplots
    for i in range(num_columns, len(ax)):
        fig.delaxes(ax[i])
    
    plt.subplots_adjust(hspace=0.2)

    plt.show()


if __name__ == '__main__':
    pass
