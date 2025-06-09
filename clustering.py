import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)

def load_data(path):
    return pd.read_csv(path)

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
    while curr_centroids != prev_centroids:
        prev_centroids = curr_centroids
        labels = assign_to_clusters(data, prev_centroids)
        curr_centroids = recompute_centroids(data, labels, k)
    centroids = curr_centroids
    return labels, centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    pass
    # plt.savefig(path)


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
    labels = numpy.array([closest_centroid(data[i],centroids) for i in range(data)])
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

