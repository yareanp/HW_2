import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.00001^2)
    """
    noise = np.random.normal(loc=0, scale=1e-5, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def min_max_scaling(nump):
    """
    scales a nump array between 0 and 1
    :param nump: dataset as numpy array of shape (n, 2)
    :return: array of shape (n, 2) scaled between 0 and 1
    """
    mins = np.min(nump, axis=0)
    maxs = np.max(nump, axis=0)
    return np.array((nump - mins) / (maxs-mins))



def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    transformed_data = add_noise(min_max_scaling(df[[features[0], features[1]]].to_numpy()))
    return transformed_data


def centroids_changed(curr_centroids, prev_centroids,k):
    """
    checks if the centroids are equal
    :param curr_centroids: numpy array of shape (n, 2)
    :param prev_centroids: numpy array of shape (n, 2)
    :param k: number of centroids
    :return: true if all centroids in curr_centroids are equal to prev_centroids
    """
    for i in range(k):
        if curr_centroids[i][0] != prev_centroids[i][0] or curr_centroids[i][1] != prev_centroids[i][1]:
            return False
    return True

def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """

    prev_centroids = choose_initial_centroids(data, k)
    labels = assign_to_clusters(data, prev_centroids)
    curr_centroids = recompute_centroids(data, labels, k)
    while not centroids_changed(curr_centroids, prev_centroids,k):
        prev_centroids = curr_centroids
        labels = assign_to_clusters(data, prev_centroids)
        curr_centroids = recompute_centroids(data, labels, k)

    centroids = curr_centroids
    return labels, centroids


def get_cmap(n, name='hsv'): #Func from Stack Overflow, just using it to generate colors
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def to_color(num):
    '''Used to set the colors of k_means graphs'''
    cmap = get_cmap(5)
    return cmap(num)


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    df = pd.DataFrame(data)
    df['labels'] = labels
    df['color'] = df['labels'].apply(to_color)
    all_data = df.to_numpy()
    plt.scatter(all_data[:, 0], all_data[:, 1], c=df['color'])
    plt.title(f'Results for kmeans with k = {len(centroids)}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X')
    plt.savefig(path)


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    sum = 0;
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2
    distance = np.sqrt(sum)
    return distance

def closest_centroid(loc, centroids):
    """
    for a speicific datapoint, finds the closest centroid
    :param loc: location of the datapoint
    :param centroids: all centroids
    :return: closest centroid to the datapoint
    """
    mincent = 0
    mindist = dist(loc, centroids[0])
    for i in range(1, len(centroids)):
        if dist(loc, centroids[i]) < mindist:
            mindist = dist(loc, centroids[i])
            mincent = i
    return mincent


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    labels = numpy.array([closest_centroid(data[i],centroids) for i in range(len(data))])
    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    df = pd.DataFrame(data)
    df['labels'] = labels #no way for a centroid to not have any matching data points, so no need to adress it
    centroids = numpy.array(df.groupby('labels').mean())
    return centroids

