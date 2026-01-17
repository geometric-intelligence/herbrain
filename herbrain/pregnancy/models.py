from functools import lru_cache
import copy

from polpo.models import DictMeshes2Comps, Meshes2Comps, ObjectRegressor
from polpo.preprocessing import Map
from polpo.preprocessing.mesh.transform import AffineTransformation
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from polpo.models import Model
from polpo.plot.mri import MriSlicer
import numpy as np


class CachingMeshModel:
    """Wrapper that precomputes and caches mesh predictions for common input values."""
    
    def __init__(self, model, precompute_range=None, precompute_values=None):
        """
        Parameters
        ----------
        model : fitted sklearn-like model
            The underlying model with predict() method
        precompute_range : tuple, optional
            (min, max, step) for precomputing predictions
        precompute_values : list, optional
            Specific values to precompute
        """
        self.model = model
        self._cache = {}
        
        # Precompute predictions for common values
        if precompute_range is not None:
            min_val, max_val, step = precompute_range
            values = np.arange(min_val, max_val + step, step)
            self._precompute(values)
        elif precompute_values is not None:
            self._precompute(precompute_values)
    
    def _precompute(self, values):
        """Precompute predictions for given values."""
        for v in values:
            key = float(round(v, 2))
            try:
                self._cache[key] = self.model.predict(np.array([[v]]))[0]
            except Exception:
                pass  # Skip values that cause errors
    
    def fit(self, X, y):
        """Fit the underlying model and precompute predictions."""
        self.model.fit(X, y)
        # Precompute for gestational weeks 0-45
        self._precompute(range(0, 46))
        return self
    
    def predict(self, X):
        """Predict with caching - returns cached result if available."""
        if hasattr(X, '__iter__') and not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # For single scalar input, check cache
        if X.shape == (1, 1):
            key = float(round(X[0, 0], 2))
            if key in self._cache:
                return [self._cache[key]]
        
        # Fall back to model prediction
        result = self.model.predict(X)
        
        # Cache the result for single inputs
        if X.shape == (1, 1):
            key = float(round(X[0, 0], 2))
            self._cache[key] = result[0]
        
        return result
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying model."""
        return getattr(self.model, name)


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
        if len(X) != 3:
            raise ValueError("Input X must be a tuple/list of (gest_week, view_index, slice_index)")
        
        gest_week, view_index, slice_index = X
        
        # Convert string view_index to int if needed
        if isinstance(view_index, str):
            view_index = {"sagittal": 0, "coronal": 1, "axial": 2}.get(view_index, 0)
        
        # Get data index from precomputed mapping
        gest_week = int(gest_week)
        if gest_week in self._gest_week_to_data_idx:
            data_idx = self._gest_week_to_data_idx[gest_week]
        else:
            data_idx = 0
        
        slice_index = int(slice_index)
        
        # Check cache first
        cache_key = (data_idx, view_index, slice_index)
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]
        
        # Compute and cache
        datum = self.data[data_idx]
        shape = datum.shape
        slice_indices = [shape[i] // 2 if i != view_index else slice_index for i in range(3)]
        slices = self.slicer.slice(datum, slice_indices)
        result = slices[view_index] if isinstance(slices, list) else slices
        
        # Cache it (limit cache size to prevent memory issues)
        if len(self._slice_cache) < 5000:
            self._slice_cache[cache_key] = result
        
        return result
        

class ClosestImageLookup(Model):
    def __init__(self, data, tar=0):
        """Model that predicts the closest image based on week number.
        Parameters:
        ----------
        data: list or array-like, paths to images for different weeks
        tar: int, 'target index' for the image (default is 0, which is the first image)
        """
        super().__init__()
        self.data = data
        self.tar = tar
        
        # Precompute week indices from image paths
        import re
        self.week_indices = []
        for path in self.data:
            match = re.search(r"week_(\d{2})", path)
            if match:
                self.week_indices.append(int(match.group(1)))
            else:
                self.week_indices.append(None)
        
        # Convert to numpy array for faster operations (filter out None values)
        valid_indices = [(i, w) for i, w in enumerate(self.week_indices) if w is not None]
        if valid_indices:
            self._valid_data_indices = np.array([i for i, _ in valid_indices])
            self._valid_weeks = np.array([w for _, w in valid_indices])
        else:
            self._valid_data_indices = np.array([0])
            self._valid_weeks = np.array([0])
        
        # Precompute week-to-image mapping for common weeks
        self._week_to_image = {}
        for week in range(0, 45):  # Typical pregnancy range
            closest_idx = self._valid_data_indices[np.argmin(np.abs(self._valid_weeks - week))]
            self._week_to_image[week] = self.data[closest_idx]

    def predict(self, X):
        """Fast prediction using precomputed mapping."""
        week = int(X[0])
        
        # Fast path: use precomputed mapping
        if week in self._week_to_image:
            return self._week_to_image[week]
        
        # Fallback: compute closest match
        closest_idx = self._valid_data_indices[np.argmin(np.abs(self._valid_weeks - week))]
        return self.data[closest_idx]