# KMeans

The project has the implementation for K-Means clustering algorithm.

Use the file - kmeans.py for generating clusters.

Sample usage:
````
python3 kmeans.py -database_file=data1.txt -k=10 -max_iters=20 -eps=0.01 -output_file=output_osid.txt
````
The file kmeans.py takes the following arguments:
1. database_file - it is the path to the database file which contains the list of all the points
2. k - it is the number of clusters
3. output_file - it is the path to store the list of the points which belong to a cluster
4. max_iters - number of maximum iterations to be run to get the cluster
5. eps - this takes the threshold which is the minimum distance you allow behind new and old centroids

Reading the data:
----------------
Used a numpy array to store the points, the dimension of the numpy array is - num_of_data_points * dimension_of_data
Used the numpy's method - ```genfromtxt(path)``` to read the data from the data file.

Generating K-Means Algorithm:
---------------------------
Important data structures are in gen_k_means method is:
1. k_mean_cluster_centroids - it is a dictionary which stores the mapping of cluster_index to it's particular centroid.
2. final_cluster_indices - it is a dictionary which stores the mapping from cluster index to list of points the respective cluster contains.
3. unchanged_centroids - it is a set of which gives information about the set of indices whose centroid have not changed, this set is useful for optimizing the number of distance measurements
4. distance_matrix - it is n * k matrix, where n is number of data points and k is number of clusters, this matrix is used to store the distances from each point to the every cluster centroid. Useful in caching the previous distance measurement if the centroid did not change.
5. clusters - it is a dictionary, this is used in intermediate steps while assigning the points to the cluster. It is mapping between from cluster index to list of points.
6. clusters_indices - it is a dictionary, this is used in intermediate steps while assigning the points to the cluster. It is mapping between from cluster index to list of indices of the points.

Important note:
python3 should be used for the program to run successfully

