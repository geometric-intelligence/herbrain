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
    def __init__(self, data, index_tar=1, slicer=None):
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

        # Get the MRI volume for the selected gestational week
        datum = self.data[gest_week - self.index_tar]

        # Use the slicer to extract the correct slice
        # The slicer expects a list of slice indices for each axis, so we build that:
        # Only the selected axis gets the slice_index, others get a default (e.g. center)
        shape = datum.shape
        slice_indices = []
        for i in range(3):
            if i == view_index:
                slice_indices.append(slice_index)
            else:
                # Use the center slice for non-selected axes
                slice_indices.append(shape[i] // 2)
        # The slicer returns all three views, but we only want the selected one
        slices = self.slicer.slice(datum, slice_indices)
        # If slicer returns a list, pick the one corresponding to view_index
        if isinstance(slices, list):
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