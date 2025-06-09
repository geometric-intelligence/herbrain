from polpo.models import DictMeshes2Comps, Meshes2Comps, ObjectRegressor
from polpo.preprocessing import Map
from polpo.preprocessing.mesh.transform import AffineTransformation
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from polpo.models import Model
from polpo.plot.mri import MriSlicer


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
        """
        data: list or array of MRI volumes (e.g. 3D numpy arrays)
        index_tar: offset for gestational week slider (usually 1)
        slicer: instance of MriSlicer (optional)
        """
        if slicer is None:
            slicer = MriSlicer()
        self.data = data
        self.index_tar = index_tar
        self.slicer = slicer
        self.hormones_df = hormones_df  # Store hormones data if needed for future use

    @classmethod
    def from_index_ordering(cls, data, index_tar=1, index_ordering=(0, 1, 2)):
        slicer = MriSlicer(index_ordering=index_ordering)
        return cls(data, index_tar, slicer)

    def predict(self, X):
        """
        X: tuple or list of (gest_week, view_index, slice_index)
           - gest_week: int, gestational week (from slider)
           - view_index: int, which view to return (from radiobutton: 0=sagittal, 1=coronal, 2=axial)
           - slice_index: int, which slice along the selected axis
        Returns: 2D numpy array, the selected MRI slice
        """
        if len(X) == 3:
            gest_week, view_index, slice_index = X
        else:
            raise ValueError("Input X must be a tuple/list of (gest_week, view_index, slice_index)")
        
        if view_index in [0, 1, 2]:
            # view_index is already an integer, use it directly
            pass
        elif view_index == "sagittal":
            view_index = 0
        elif view_index == "coronal":
            view_index = 1
        elif view_index == "axial":
            view_index = 2
        else:
            raise ValueError(f"view_index: {view_index}, must be 'sagittal', 'coronal', or 'axial'")
        
        gest_week_id = "gestWeek"  # This is the column name in hormones_df for gestational week
        # Use hormones_df to compute the session number associated with the gestational week.
        # We assume hormones_df is indexed by session or has a column for gestational week.
        # Try to find the session number corresponding to the given gest_week.
        # If hormones_df is a DataFrame with a 'gest_week' column, find the session index.
        session_number = None
        if hasattr(self.hormones_df, "loc") and gest_week_id in self.hormones_df.columns:
            # Find the first row where gest_week matches
            matches = self.hormones_df[self.hormones_df[gest_week_id] == gest_week]
            if not matches.empty:
                # Use the index of the first match as the session number
                session_number = matches.index[0]
            else:
                # If not found, return the closest match
                # Find the closest gestational week in the DataFrame
                all_gest_weeks = self.hormones_df[gest_week_id]
                closest_idx = (all_gest_weeks - gest_week).abs().idxmin()
                session_number = closest_idx

        if session_number is not None and session_number >= len(self.data): # address the debug mode.
            session_number = len(self.data) - 1
        # Get the MRI volume for the selected gestational week
        datum = self.data[session_number - self.index_tar] # this needs to be session number, not gest week.

        # Use the slicer to extract the correct slice
        # The slicer expects a list of slice indices for each axis, so we build that:
        # Only the selected axis gets the slice_index, others get a default (e.g. center)
        shape = datum.shape
        print(f"Datum shape: {shape}, view_index: {view_index}, slice_index: {slice_index}")
        slice_indices = []
        for i in range(3):
            if i == view_index:
                slice_indices.append(slice_index)
            else:
                # Use the center slice for non-selected axes
                slice_indices.append(shape[i] // 2)
        # The slicer returns all three views, but we only want the selected one
        slices = self.slicer.slice(datum, slice_indices)
        print(f"len(slices): {len(slices)}, slice_indices: {slice_indices}")
        # If slicer returns a list, pick the one corresponding to view_index
        if isinstance(slices, list):
            print(f"Returning slice for view_index {view_index}: {slices[view_index]}")
            return slices[view_index]
        else:
            return slices
        

class ClosestImageLookup(Model):
    def __init__(self, data, tar=0):
        super().__init__()
        self.data = data
        self.tar = tar
        # Precompute week indices from image paths
        self.week_indices = []
        for path in self.data:
            # Extract the week index from the path, assuming format contains "week_{index:02}"
            import re
            match = re.search(r"week_(\d{2})", path)
            if match:
                self.week_indices.append(int(match.group(1)))
            else:
                self.week_indices.append(None)  # or raise an error if strict

    def predict(self, X):
        # Expects X to be a tuple/list with the week number as the first element
        week = X[0]
        # Find the closest week index
        min_diff = float("inf")
        closest_idx = 0
        for i, w in enumerate(self.week_indices):
            if w is not None:
                diff = abs(w - week)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
        return self.data[closest_idx]