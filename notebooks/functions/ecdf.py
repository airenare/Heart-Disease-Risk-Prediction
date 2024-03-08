import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_ecdf(data, x_label=None, normal_overlay=True):
    """
    Plots the Empirical Cumulative Distribution Function (ECDF) of a dataset.
    Args:
        data (pandas Series): Data to be plotted
        x_label (str): Label for the x-axis (optional)
        normal_overlay (bool): Whether to overlay a normal distribution on the ECDF (optional)
    """

    # Number of data points
    n = len(data)

    # x-data for the ECDF
    x = np.sort(data)

    # y-data for the ECDF
    y = np.arange(1, n + 1) / n

    # Plot the ECDF
    plt.plot(x, y, marker='.', linestyle='none')

    # Label the axes
    plt.xlabel(x_label) if x_label else plt.xlabel('x')
    plt.ylabel('ECDF')

    # Overlay a normal distribution if requested
    if normal_overlay:
        # Compute mean and standard deviation
        mu = np.mean(data)
        sigma = np.std(data)

        # Sample out of a normal distribution with the same mean and standard deviation
        samples = np.random.normal(mu, sigma, size=10000)

        # Get the CDF of the samples and of the data
        x_theo = np.sort(samples)
        y_theo = np.arange(1, len(x_theo) + 1) / len(x_theo)

        # Plot the CDFs
        plt.plot(x_theo, y_theo)

    # Display the plot
    plt.show()


def plot_ecdf_overlay(data, x_label=None, overlay=None, axis=None):
    """
    Plots the Empirical Cumulative Distribution Function (ECDF) of a dataset.
    Args:
        data (pandas Series): Data to be plotted
        x_label (str): Label for the x-axis (optional)
        overlay (str): Type of distribution to overlay on the ECDF (optional)
        axis (matplotlib.axes._subplots.AxesSubplot): Axis for plotting (optional)
    """

    # Number of data points
    n = len(data)

    # x-data for the ECDF
    x = np.sort(data)

    # y-data for the ECDF
    y = np.arange(1, n + 1) / n

    # Plot the ECDF
    if axis is None:
        fig, axis = plt.subplots()

    axis.plot(x, y, marker='.', linestyle='none', label='ECDF')

    # Overlay a normal distribution if requested
    # Compute mean and standard deviation
    mu = np.mean(data)
    sigma = np.std(data)

    if overlay is None:
        pass

    elif overlay == 'normal':
        # Sample out of a normal distribution with the same mean and standard deviation
        samples = np.random.normal(mu, sigma, size=10000)

        # Get the CDF of the samples and of the data
        x_theo = np.sort(samples)
        y_theo = np.arange(1, len(x_theo) + 1) / len(x_theo)



    elif overlay == 'exponential':
        # If mean is negative, adjust it
        mu = 0.0000001 if mu < 0 else mu

        # Sample out of an exponential distribution with the same mean and standard deviation
        samples = np.random.exponential(1/mu, size=len(data))

        # Get the CDF of the samples and of the data
        x_theo = np.sort(samples)
        y_theo = np.arange(1, len(x_theo) + 1) / len(x_theo)


    elif overlay == 'uniform':
        # Sample out of a uniform distribution with the same mean and standard deviation
        samples = np.random.uniform(low=data.min(), high=data.max(), size=10000)

        # Get the CDF of the samples and of the data
        x_theo = np.sort(samples)
        y_theo = np.arange(1, len(x_theo) + 1) / len(x_theo)


    elif overlay == 'binomial':
        # If mean is negative or greater than 1, adjust it
        if mu < 0:
            mu = 0.0000001
        elif mu > 1:
            mu = 0.9999999


        n = 1
        p = mu / n

        # Sample out of a binomial distribution with the same mean and standard deviation
        samples = np.random.binomial(n=n, p=p, size=len(data))

        # Get the CDF of the samples and of the data
        x_theo = np.sort(samples)
        y_theo = np.arange(1, len(x_theo) + 1) / len(x_theo)


    else:
        raise ValueError('Invalid distribution type')

    # print(f'{mu = }')

    # Plot the CDFs
    axis.plot(x_theo, y_theo, label=overlay) if overlay is not None else None


    # Label the axes
    axis.set_xlabel(x_label) if x_label else axis.set_xlabel('x')
    axis.set_ylabel('ECDF')

    # Display legend if overlay is specified
    if overlay:
        axis.legend()

    # Display the plot if not already done within the overlay code
    if axis is None:
        plt.show()


def testing(dist_type='normal'):
    """
    Test function for the ECDF plotter.
    Args:
        dist_type (str): Type of distribution to test (normal, exponential, uniform, binomial)
    """
    # Seed random number generator
    np.random.seed(42)

    match dist_type:
        case 'normal':
            # Random numbers from a normal distribution
            test_data = np.random.normal(loc=0, scale=1, size=1000)
        case 'exponential':
            # Random numbers from an exponential distribution
            test_data = np.random.exponential(scale=1, size=1000)
        case 'uniform':
            # Random numbers from a uniform distribution
            test_data = np.random.uniform(low=0, high=1, size=1000)
        case 'binomial':
            # Random numbers from a binomial distribution
            test_data = np.random.binomial(n=100, p=0.05, size=1000)
        case _:
            raise ValueError('Invalid distribution type')

    # Plot the ECDF
    plot_ecdf(test_data, x_label='Value', normal_overlay=True)


def test_all_distributions(data):
    distributions = ['normal', 'exponential', 'uniform', 'binomial']

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i, distribution in enumerate(distributions):
        ax = axes[i // 2, i % 2]
        plot_ecdf_overlay(data, overlay=distribution, axis=ax)
        ax.set_title(f'ECDF with {distribution} Overlay')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':



    # Random numbers from a normal distribution
    data1 = np.random.normal(loc=0, scale=1, size=1000)

    # Random numbers from an exponential distribution
    data2 = np.random.exponential(scale=1, size=1000)

    # Random numbers from a uniform distribution
    data3 = np.random.uniform(low=0, high=1, size=1000)

    # Random numbers from a binomial distribution
    data4 = np.random.binomial(n=5, p=0.05, size=1000)

    # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # plot_ecdf_overlay(dt, overlay='exponential', ax=ax[1, 0])
    # plt.show()

    for d in [data1, data2, data3, data4]:
        test_all_distributions(d)
