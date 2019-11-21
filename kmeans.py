import argparse
import os
import numpy as np
import math


# method to check whether the points are converged or not
def is_converged(_k_mean_cluster, _new_k_mean_cluster, _eps, _iteration_number):
    # this is a base if iteration number is 0 then return false
    if _iteration_number == 0:
        return False

    _max_dist = -math.inf
    # looping through cluster centroids and checking the distance of between new and old centroid
    for i, _old_centroid in _k_mean_cluster.items():
        _new_centroid = _new_k_mean_cluster[i]
        # finding the distance between old and new centroid
        _dist = np.linalg.norm(_old_centroid - _new_centroid)
        _max_dist = max(_max_dist, _dist)

    return _max_dist <= _eps


# method to generate clusters using k-means algorithm
def gen_k_means(data, k, max_iterations, eps, output_file_path):

    # seeding the value for random indices generation
    np.random.seed(100)

    # getting the random k indices
    centroids_indices = np.random.randint(len(data), size=k)

    # dictionary to map between cluster index and centroids
    k_mean_cluster_centroids = dict()
    for i in range(0, k):
        # filling the dictionary by mapping the cluster index to initial centroids
        k_mean_cluster_centroids[i] = data[centroids_indices[i]]

    new_k_mean_cluster_centroids = k_mean_cluster_centroids
    iteration_number = 0
    # dictionary to store the final indices for each cluster
    final_cluster_indices = dict()
    # set to have unchanged centroids
    unchanged_centroids = set()
    # distance matrix between each point and cluster
    distance_matrix = np.zeros((data.shape[0], k))

    # while loop to generate and assign the cluster
    while not is_converged(k_mean_cluster_centroids, new_k_mean_cluster_centroids, eps, iteration_number) and max_iterations > iteration_number:

        k_mean_cluster_centroids = dict(new_k_mean_cluster_centroids)

        # dictionary to store the mapping from cluster index to list of points
        clusters = dict()
        # dictionary to store the mapping from cluster index to list of point indices
        clusters_indices = dict()

        """ Assign  each  data  point  to  each  of  the k  clusters  based on  Euclidean  distance """
        # looping through each data point
        for point_index, point in enumerate(data):
            # for each data point calculating the distance fro each centroid
            for i, centroid in k_mean_cluster_centroids.items():
                # only calculating the distances for the centroids which are changed
                # and for the unchanged centroid the values in the distance matrix remains same
                if i not in unchanged_centroids:
                    distance = np.linalg.norm(point - centroid)
                    distance_matrix[point_index][i] = distance

            # getting the cluster index of smallest distance
            cluster_index = int(np.argmin(distance_matrix[point_index]))

            # getting the point list for that specific cluster index
            cluster_list = clusters.get(cluster_index, [])
            # appending the point to the list
            cluster_list.append(point)
            # adding back the list to the dictionary
            clusters[cluster_index] = cluster_list

            # getting the list of indices of points for that specific cluster index
            cluster_index_list = clusters_indices.get(cluster_index, [])
            # appending the index to the list
            cluster_index_list.append(point_index)
            # adding back the list to the dictionary
            clusters_indices[cluster_index] = cluster_index_list

        # resetting the unchanged centroids
        unchanged_centroids = set()

        """ updating  the  cluster  centroids """
        # looping through each key and each list in the dictionary
        for _cluster_index, cluster_list in clusters.items():
            # converting the list to numpy array
            np_cluster_list = np.asarray(cluster_list)
            # getting the mean
            new_centroid = np.mean(np_cluster_list, axis=0)
            # adding to the unchanged_centroids if both the previous and new centroids are same
            if np.array_equal(new_centroid, k_mean_cluster_centroids[_cluster_index]):
                unchanged_centroids.add(_cluster_index)
            # updating the new centroid
            new_k_mean_cluster_centroids[_cluster_index] = new_centroid

        # updating the final cluster indices
        final_cluster_indices = clusters_indices
        iteration_number += 1

    """ outputting the cluster centroids """
    # opening the file path
    f = open(output_file_path, mode='w')

    # looping through each cluster index
    for i in range(0, k):
        # getting the list of the indices
        indices_list = sorted(final_cluster_indices[i])
        f.write(str(i) + ': ')
        f.write(' '.join([str(elem) for elem in indices_list]))
        f.write('\n')

    f.close()


# reading the data from the data_base file
def read_data(data_file):
    # using numpy method to read the data
    data = np.genfromtxt(data_file).astype(float)
    return data


# method to get the parser
def get_parser():
    _parser = argparse.ArgumentParser()

    # argument to get take the path of database file
    _parser.add_argument('-database_file')

    # argument to get the number of clusters
    _parser.add_argument('-k')

    # argument to get maximum number of iterations
    _parser.add_argument('-max_iters')

    # argument to get the value of eps
    _parser.add_argument('-eps')

    # argument to get take the path of output file
    _parser.add_argument('-output_file')

    return _parser


def main():
    # getting the parser
    parser = get_parser()
    args = parser.parse_args()

    # get all the arguments required for the program to run
    database_file = os.path.abspath(str(args.database_file))
    k = int(str(args.k))
    max_iters = int(str(args.max_iters))
    eps = float(str(args.eps))
    output_file = os.path.abspath(str(args.output_file))

    # reading the database
    data = read_data(database_file)

    # generating k clusters
    gen_k_means(data, k, max_iters, eps, output_file)


# main method where the program starts
if __name__ == '__main__':
    # see the main method for details
    main()
