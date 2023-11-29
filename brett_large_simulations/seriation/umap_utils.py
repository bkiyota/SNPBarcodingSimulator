import pynndescent
import scipy.sparse
import numpy as np
import scipy.sparse
import warnings


# Fake class to pass to UMAP when feeding it nearest neighbors
class MyNNDescent(pynndescent.NNDescent):
    def __init__(self):
        return


def nearest_neighbors_from_neighbor_graph(G, n_neighbors=None, D=None):
    """
    Find the ``n_neighbors`` nearest points for each data point in ``X``
    in the similarity matrix ``G``.
    Returns a tuple that can be fed to UMAP using ``precomputed_knn``,
    in the same way as the nearest_neighbors() function does it.
    If ``n_neighbors`` is None, this function returns the largest number of neighbors
    (``max_neighbors``) it can find in the neighbor graph ``G`` (same number for all points).
    Else, it returns the minimum of ``n_neighbors`` and ``max_neighbors``.
    If ``D`` is None, the distances to neighbors is computed as 1-sim, where sim is the
    similarity found in the neighbor graph ``G``.
    Else, it will return the corresponding distances found in D.

    Parameters
    ----------
    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    G: sparse or dense array of shape (n_samples, n_samples)
        The (sparse) similarity matrix between samples. Diagonal elements are ignored.

    D: array of shape (n_samples, n_samples)
        The all-pair distance matrix between samples.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    nnd: 'fake' NNDescent object
        UMAP requires this object, but only uses it when embedding new data (using transform())
        or updating the current embedding with new data (using update()).
        If you're not using any of these 2 functions, this fake object will be enough.
    """

    N = G.shape[0]

    # If G is a np.matrix, convert it to an array
    if isinstance(G, np.matrix): G = np.asarray(G)

    # Remove non-zero diagonal elements if any, as they are not relevant.
    if np.any(G.diagonal() > 0):
        G = np.asarray(G - scipy.sparse.diags(G.diagonal(), 0, [N, N]))

    # Get lists of neighbor indices and similarities
    if scipy.sparse.issparse(G):
        neighbor_similarities = [G.data[G.indptr[i]:G.indptr[i+1]] for i in range(N)]
        neighbor_indices = [G.indices[G.indptr[i]:G.indptr[i+1]] for i in range(N)]
    else:
        neighbor_similarities = [G[i][G[i] > 0] for i in range(N)]
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(N)]

    # Get maximum number of neighbors from graph
    max_neighbors = np.min([len(l) for l in neighbor_indices])
    if max_neighbors == 0:
        print("ERROR: Some points in the graph have 0 neighbors.")
        exit(-1)
    if not n_neighbors:
        n_neighbs = max_neighbors
    else:
        if max_neighbors < n_neighbors:
            warnings.warn("The neighbor graph has a minimum of %d nearest neighbors. "
                  "This function will return that many instead of the given n_neighbors=%d."
                  %(max_neighbors,n_neighbors))
            n_neighbors = max_neighbors
        n_neighbs = min(max_neighbors,n_neighbors)

    # Find the k nearest neighbors in the sense of the similarity G, not of the distance in D
    knn_indices = np.vstack([neighbor_indices[i][np.argsort(neighbor_similarities[i])[:-1-n_neighbs:-1]] for i in range(N)])
    # Get the distances to these knn.
    dist_ind_i = np.arange(N).repeat(n_neighbs)
    dist_ind_j = knn_indices.flatten()

    if D is not None:
        knn_dists = D[dist_ind_i,dist_ind_j].reshape(N, n_neighbs)               # Return real distances
    else:
        knn_dists = 1-np.array(G[dist_ind_i,dist_ind_j]).reshape(N, n_neighbs)    # Return distances based on similarity

    nnd = MyNNDescent()

    return knn_indices, knn_dists, nnd


