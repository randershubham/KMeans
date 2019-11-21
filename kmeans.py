import argparse
import os
import numpy as np
import math


def is_converged(_k_mean_cluster, _new_k_mean_cluster, _eps, _iteration_number):
    if _iteration_number == 0:
        return False
    _max_dist = -math.inf
    for i, _old_centroid in _k_mean_cluster.items():
        _new_centroid = _new_k_mean_cluster[i]
        _dist = np.linalg.norm(_old_centroid - _new_centroid)
        _max_dist = max(_max_dist, _dist)

    return _max_dist <= _eps


def gen_k_means(data, k, max_iterations, eps, output_file_path):
    np.random.seed(100)
    centroids_indices = np.random.randint(len(data), size=k)

    k_mean_cluster_centroids = dict()
    for i in range(0, k):
        k_mean_cluster_centroids[i] = data[centroids_indices[i]]

    distance_matrix = np.zeros((data.shape[0], k))

    new_k_mean_cluster_centroids = k_mean_cluster_centroids

    iteration_number = 0

    final_cluster_indices = dict()
    unchanged_centroids = set()

    while not is_converged(k_mean_cluster_centroids, new_k_mean_cluster_centroids, eps, iteration_number) and max_iterations > iteration_number:
        k_mean_cluster_centroids = dict(new_k_mean_cluster_centroids)

        clusters = dict()
        clusters_indices = dict()

        for point_index, point in enumerate(data):

            for i, centroid in k_mean_cluster_centroids.items():
                if i not in unchanged_centroids:
                    distance = np.linalg.norm(point - centroid)
                    distance_matrix[point_index][i] = distance

            cluster_index = int(np.argmin(distance_matrix[point_index]))

            cluster_list = clusters.get(cluster_index, [])
            cluster_list.append(point)
            clusters[cluster_index] = cluster_list

            cluster_index_list = clusters_indices.get(cluster_index, [])
            cluster_index_list.append(point_index)
            clusters_indices[cluster_index] = cluster_index_list

        unchanged_centroids = set()

        for _cluster_index, cluster_list in clusters.items():
            np_cluster_list = np.asarray(cluster_list)
            new_centroid = np.mean(np_cluster_list, axis=0)
            if np.array_equal(new_centroid, k_mean_cluster_centroids[_cluster_index]):
                unchanged_centroids.add(_cluster_index)
            new_k_mean_cluster_centroids[_cluster_index] = new_centroid

        final_cluster_indices = clusters_indices
        iteration_number += 1

    f = open(output_file_path, mode='w')
    for i, indices_list in final_cluster_indices.items():
        f.write(str(i) + ': ')
        f.write(' '.join([str(elem) for elem in indices_list]))
        f.write('\n')

    f.close()


def read_data(data_file):
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
    parser = get_parser()
    args = parser.parse_args()

    # get all the arguments required for the program to run
    database_file = os.path.abspath(str(args.database_file))
    k = int(str(args.k))
    max_iters = int(str(args.max_iters))
    eps = float(str(args.eps))
    output_file = os.path.abspath(str(args.output_file))

    data = read_data(database_file)

    gen_k_means(data, k, max_iters, eps, output_file)


# main method where the program starts
if __name__ == '__main__':
    # see the main method for details
    main()
