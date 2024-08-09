"""Functions to smoothen the vertices of a mesh using different techniques."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_neighbors(mesh, k=5):
    """Compute the k-nearest neighbors for each vertex in a mesh based on 3D spatial data.

    Parameters:
        mesh (np.ndarray): Array of shape (n_vertices, 3) containing the vertices of the mesh.
        k (int): Number of neighbors to find for each vertex.

    Returns:
        list of lists: Each sublist contains the indices of the k nearest neighbors for each vertex.
    """
    # Initialize NearestNeighbors model
    neigh = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(mesh)

    # Find k+1 nearest neighbors (including the point itself)
    _, indices = neigh.kneighbors(mesh)

    # Create a list of lists for neighbor indices, excluding the point itself from its list of neighbors
    neighbors = [
        list(ind[1:]) for ind in indices
    ]  # skip the first one as it is the point itself

    return neighbors


def median_smooth(mesh, neighbors):
    """Smooth a mesh using the median coordinates of neighboring vertices.

    Parameters
    ----------
    mesh (np.ndarray): Array of shape (n_vertices, 3) containing the vertices of the mesh.
    neighbors (list of lists): Each sublist contains the indices of the neighboring vertices for each vertex.

    Returns
    -------
    np.ndarray: Array of shape (n_vertices, 3) containing the smoothed vertices of the mesh.
    """
    n_vertices, _ = mesh.shape
    mesh_smooth = np.zeros_like(mesh)

    for i in range(n_vertices):
        if len(neighbors[i]) == 0:
            # If no neighbors, do not change the vertex
            mesh_smooth[i] = mesh[i]
        else:
            # Gather all neighboring vertices
            neighbor_vertices = mesh[neighbors[i]]
            # Compute the median along each coordinate
            mesh_smooth[i] = np.median(neighbor_vertices, axis=0)

    return mesh_smooth
