from polpo.models import DictMeshes2Comps, Meshes2Comps, Model, ObjectRegressor
from polpo.preprocessing import Map
from polpo.preprocessing.mesh.transform import AffineTransformation
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from polpo.plot.mri import MriSlicer
import numpy as np


def MeshPCR(model=None, affine_transform=None, n_components=4, n_pipes=None):
    """Linear regression on PCA components of transformed meshes."""
    # n_pipes: if dict with multiple structures

    if model is None:
        model = LinearRegression()

    mesh_transform = (
        Map(step=[AffineTransformation(transform=affine_transform)])
        if affine_transform is not None
        else None
    )

    if n_pipes is None:
        objs2y = Meshes2Comps(
            dim_reduction=PCA(n_components=n_components),
            smoother=False,
            mesh_transform=mesh_transform,
        )
    else:
        objs2y = DictMeshes2Comps(
            dim_reduction=PCA(n_components=n_components),
            smoother=False,
            mesh_transform=mesh_transform,
            n_pipes=n_pipes,
        )

    return ObjectRegressor(model=model, objs2y=objs2y)


class MriModel(Model):
    def __init__(self, data, hormones_df, index_tar=1, slicer=None):
        """Model for predicting MRI slices based on gestational week and view index.
        
        OPTIMIZED: Uses efficient caching for fast lookups.

        Parameters:
        ----------
        data: list or array-like, MRI data for different gestational weeks
        hormones_df: pandas DataFrame, contains gestational week information
        index_tar: int, 
            The number at which the target MRI data starts in the list
        slicer: MriSlicer, optional, used to slice the MRI data
        """
        if slicer is None:
            slicer = MriSlicer()
        self.data = data
        self.index_tar = index_tar
        self.slicer = slicer
        self.hormones_df = hormones_df
        self._num_data = len(data)
        
        # Precompute gestational week to data index mapping
        self._gest_week_to_data_idx = {}
        gest_week_id = "gestWeek"
        if hasattr(hormones_df, "loc") and gest_week_id in hormones_df.columns:
            all_gest_weeks = hormones_df[gest_week_id].values
            all_indices = hormones_df.index.values
            
            # Map each gestational week (0-45) to closest data index
            for week in range(0, 46):
                closest_session_idx = np.argmin(np.abs(all_gest_weeks - week))
                session = all_indices[closest_session_idx]
                data_idx = min(session - index_tar, self._num_data - 1)
                data_idx = max(0, data_idx)
                self._gest_week_to_data_idx[week] = data_idx
        
        # Cache for slices - populated on demand
        self._slice_cache = {}
        self._data_shapes = [d.shape for d in data]

    @classmethod
    def from_index_ordering(cls, data, hormones_df, index_tar=1, index_ordering=(0, 1, 2)):
        slicer = MriSlicer(index_ordering=index_ordering)
        return cls(data, hormones_df, index_tar, slicer)

    def predict(self, X):
        """Fast prediction with caching."""
        # Handle None or invalid inputs gracefully to prevent callback failures
        try:
            if X is None or (hasattr(X, '__len__') and len(X) != 3):
                # Return default empty slice if inputs are invalid
                if len(self.data) > 0:
                    sample_shape = self.data[0].shape
                    return np.zeros((sample_shape[1], sample_shape[2]), dtype=np.float32)
                return np.zeros((100, 100), dtype=np.float32)
            
            gest_week, view_index, slice_index = X
            
            # Handle None values
            if gest_week is None:
                gest_week = 15  # Default to week 15
            if view_index is None:
                view_index = 0  # Default to sagittal
            if slice_index is None:
                slice_index = 0  # Default to first slice
            
            # Convert string view_index to int if needed
            if isinstance(view_index, str):
                view_index = {"sagittal": 0, "coronal": 1, "axial": 2}.get(view_index.lower(), 0)
            
            # Get data index from precomputed mapping
            try:
                gest_week = int(gest_week)
            except (ValueError, TypeError):
                gest_week = 15  # Default to week 15
            
            if gest_week in self._gest_week_to_data_idx:
                data_idx = self._gest_week_to_data_idx[gest_week]
            else:
                data_idx = 0
            
            try:
                slice_index = int(slice_index)
            except (ValueError, TypeError):
                slice_index = 0
            
            # Ensure indices are within bounds
            if data_idx < 0 or data_idx >= len(self.data):
                data_idx = 0
            
            # Check cache first
            cache_key = (data_idx, view_index, slice_index)
            if cache_key in self._slice_cache:
                return self._slice_cache[cache_key]
            
            # Compute and cache
            datum = self.data[data_idx]
            shape = datum.shape
            
            # Ensure slice_index is within bounds for the selected view
            max_slice = shape[view_index] - 1
            if slice_index < 0:
                slice_index = 0
            elif slice_index > max_slice:
                slice_index = max_slice
            
            slice_indices = [shape[i] // 2 if i != view_index else slice_index for i in range(3)]
            slices = self.slicer.slice(datum, slice_indices)
            result = slices[view_index] if isinstance(slices, list) else slices
            
            # Cache it (limit cache size to prevent memory issues)
            if len(self._slice_cache) < 5000:
                self._slice_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            import traceback
            print(f"Error in MriModel.predict(): {e}")
            traceback.print_exc()
            # Return a default empty slice
            if len(self.data) > 0:
                sample_shape = self.data[0].shape
                return np.zeros((sample_shape[1], sample_shape[2]), dtype=np.float32)
            return np.zeros((100, 100), dtype=np.float32)