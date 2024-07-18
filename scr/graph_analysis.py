import numpy as np
from sklearn.metrics import pairwise_distances, cosine_similarity
from sklearn.neighbors import kneighbors_graph

class GraphAnalyzer:
    """
    A class to perform graph analysis including k-NN graph construction and affinity matrix computation.
    
    Attributes:
    ----------
    data : np.ndarray
        Input data for graph analysis.
    knn_graph : np.ndarray
        k-NN graph constructed from the input data.
    affinity_matrix : np.ndarray
        Affinity matrix computed from the input data.

    Methods:
    -------
    compute_adaptive_k(distances, max_k):
        Computes the adaptive k value for each data point.
    compute_adaptive_sigma_cosine(k):
        Computes the adaptive sigma value using cosine distances.
    construct_knn_graph(max_k):
        Constructs a k-NN graph with mutual k-nearest neighbors.
    compute_affinity_matrix():
        Computes the affinity matrix using cosine similarity and Pearson correlation.
    """
    
    def __init__(self, data):
        """
        Constructs all the necessary attributes for the GraphAnalyzer object.
        
        Parameters:
        ----------
        data : np.ndarray
            Input data for graph analysis.
        """
        self.data = data
    
    def compute_adaptive_k(self, distances, max_k):
        """
        Computes the adaptive k value for each data point based on the average distance.
        
        Parameters:
        ----------
        distances : np.ndarray
            Pairwise distance matrix of the input data.
        max_k : int
            Maximum number of neighbors to consider.
        
        Returns:
        -------
        np.ndarray
            Array of adaptive k values for each data point.
        """
        k_adaptive = np.zeros(distances.shape[0], dtype=int)
        for i, dist_row in enumerate(distances):
            avg_dist = np.mean(dist_row[:max_k])
            for k in range(2, max_k):
                if dist_row[k] > avg_dist:
                    k_adaptive[i] = k
                    break
            if k_adaptive[i] == 0:
                k_adaptive[i] = max_k - 1
        return k_adaptive
    
    def compute_adaptive_sigma_cosine(self, k):
        """
        Computes the adaptive sigma value using cosine distances.
        
        Parameters:
        ----------
        k : int
            Number of nearest neighbors to consider for sigma computation.
        
        Returns:
        -------
        np.ndarray
            Array of adaptive sigma values for each data point.
        """
        cosine_dist = pairwise_distances(self.data, metric='cosine')
        sorted_cosine_distances = np.sort(cosine_dist, axis=1)
        return sorted_cosine_distances[:, k]
    
    def construct_knn_graph(self, max_k=10):
        """
        Constructs a k-NN graph with mutual k-nearest neighbors.
        
        Parameters:
        ----------
        max_k : int, optional
            Maximum number of neighbors to consider (default is 10).
        
        Returns:
        -------
        np.ndarray
            k-NN graph constructed from the input data.
        """
        distances = pairwise_distances(self.data)
        k_adaptive = self.compute_adaptive_k(distances, max_k)
        self.knn_graph = kneighbors_graph(self.data, n_neighbors=max_k, mode='connectivity', include_self=True)
        return self.knn_graph.toarray()
    
    def compute_affinity_matrix(self):
        """
        Computes the affinity matrix using cosine similarity and Pearson correlation.
        
        Returns:
        -------
        np.ndarray
            Affinity matrix computed from the input data.
        """
        pearson_corr = np.corrcoef(self.data)
        cosine_sim = cosine_similarity(self.data)
        self.affinity_matrix = (cosine_sim + pearson_corr) / 2
        self.affinity_matrix = np.abs(self.affinity_matrix)
        np.fill_diagonal(self.affinity_matrix, self.affinity_matrix.diagonal() + 1e-5)
        return self.affinity_matrix
