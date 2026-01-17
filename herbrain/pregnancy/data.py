import os

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from polpo.preprocessing import (
    ListSqueeze,
    Map,
    PartiallyInitializedStep,
    Pipeline,
    Sorter,
    Truncater,
)
from polpo.preprocessing.load.pregnancy.jacobs import (
    MeshLoader as MaternalMeshLoader,
    TabularDataLoader as MaternalCsvDataLoader,
)
from polpo.preprocessing.load.pregnancy.pilot import (
    MriLoader as PilotMriLoader,
    TabularDataLoader as PilotCsvDataLoader,
)
from polpo.preprocessing.mesh.conversion import TrimeshFromData, TrimeshFromPvMesh
from polpo.preprocessing.mesh.io import PvReader, FreeSurferReader
from polpo.preprocessing.mesh.registration import PvAlign
from polpo.preprocessing.mri import (
    MriImageLoader,
    SkimageMarchingCubes,
)

# TODO: check docstrings

# TODO: Pipeline -> Pipe

# TODO: if general enough, move to polpo


class PilotMriImageLoader(Pipeline):
    """Load, sort, truncate, and parse MRI images.
    
    Parameters:
    ----------
    debug : bool, optional
        If True, uses a smaller subset of images for debugging.
    data_dir : str, optional
        Directory where MRI images are stored. If None, defaults to HERBRAIN_DATA_DIR.
        
    Returns:
    -------
    Pipeline
        A pipeline that loads, sorts, truncates, and parses MRI images.
    """

    def __init__(self, debug=False, data_dir=None):
        if debug:
            value = 2
            n_jobs = 1
            verbose = 1
        else:
            value = None
            n_jobs = -1
            verbose = 0

        super().__init__(
            steps=[
                PilotMriLoader(data_dir=data_dir, as_image=False),
                ppdict.DictToValuesList(),
                Sorter(),
                Truncater(value=value),
                Map(n_jobs=n_jobs, verbose=verbose, step=MriImageLoader()),
            ]
        )


class HormonesCsvLoader(Pipeline):
    """Load maternal hormone data and drop repeated row.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory where data is stored. 
        If the raw/28Baby_Hormones.csv file exists in this directory, it will be read directly.
        Otherwise, it will attempt to download from Figshare.
    """

    def __init__(self, data_dir=None):
        # Try to read the CSV directly if it exists to avoid permission issues
        # when the data is read-only
        if data_dir is not None:
            raw_csv_path = os.path.join(data_dir, "raw", "28Baby_Hormones.csv")
            if os.path.exists(raw_csv_path):
                # Read directly with pandas preprocessing
                super().__init__(
                    steps=[
                        lambda _: raw_csv_path,
                        ppd.CsvReader(),
                        ppd.UpdateColumnValues(
                            column_name="sessionID",
                            func=lambda entry: int(entry.split("-")[1]),
                        ),
                        ppd.DfFilter(lambda df: df["sessionID"] == 27, negate=True),
                        ppd.IndexSetter("sessionID", drop=True),
                    ]
                )
                return
        
        # Fall back to the standard loader
        super().__init__(
            steps=[
                PilotCsvDataLoader(data_dir=data_dir),
                # No need to drop row 27 since PilotCsvDataLoader already filters it
            ]
        )


class TemplateImageLoader(Pipeline):
    """Load and parse a single MRI image as a template with affine."""

    def __init__(self, data_dir=None):
        super().__init__(
            steps=[
                PilotMriLoader(data_dir=data_dir, subset=[1]),
                ppdict.DictToValuesList(),
                ListSqueeze(),
                MriImageLoader(as_nib=True),
            ]
        )


class NibImage2Mesh(Pipeline):
    """Generate a surface mesh from a 3D MRI image."""

    def __init__(self):
        super().__init__(
            steps=[
                lambda x: x.get_fdata(),
                SkimageMarchingCubes(return_values=False),
                TrimeshFromData(),
            ]
        )


class MaternalRegisteredMeshesLoader(Pipeline):
    """
    Load, align, and convert a set of per-subject meshes to trimesh format.

    The pipeline loads meshes in Pv format, aligns them to a template,
    and converts them into `trimesh.Trimesh` objects. The key is the subject/session ID.

    Parameters
    ----------
    data_dir : str, optional
        Directory where data is stored.
    max_iterations : int, optional
        Maximum number of iterations for alignment.
    derivative : str, optional
        Derivative folder (e.g. "enigma", "fsl_first"). Default is "enigma".
    subject_subset : list, optional
        Subset of subject IDs to load. If None, loads pilot subject "01".
    struct_subset : list, optional
        Subset of structure names. Default is ["L_Hipp"].
    """

    def __init__(
        self,
        data_dir=None,
        max_iterations=500,
        derivative="enigma",
        subject_subset=None,
        struct_subset=None,
    ):
        if subject_subset is None:
            subject_subset = ["01"]
        if struct_subset is None:
            struct_subset = ["L_Hipp"]

        mesh_loader_kwargs = dict(
            derivative=derivative,
            subject_subset=subject_subset,
            struct_subset=struct_subset,
            as_mesh=True,
        )
        if data_dir is not None:
            mesh_loader_kwargs["data_dir"] = data_dir

        super().__init__(
            steps=[
                MaternalMeshLoader(**mesh_loader_kwargs),
                # Extract unique keys to flatten nested dict structure
                ppdict.ExtractUniqueKey(nested=True),
                PartiallyInitializedStep(
                    Step=lambda target: ppdict.DictMap(
                        PvAlign(target=target, max_iterations=max_iterations)
                    ),
                    _target=lambda meshes: meshes[list(meshes.keys())[0]],
                ),
                ppdict.DictMap(step=TrimeshFromPvMesh()),
            ]
        )


class MultipleMaternalMeshesLoader(Pipeline):
    """
    Load, align, and convert Pv meshes for a list of brain structures.

    This pipeline loads meshes using the new `MeshLoader` API and applies
    joint registration + conversion to Trimesh format.

    Parameters
    ----------
    data_dir : str, optional
        Directory where data is stored.
    max_iterations : int, optional
        Maximum iterations for PvAlign. Default is 500.
    derivative : str, optional
        Derivative folder (e.g. "enigma", "fsl_first"). Default is "enigma".
    subject_subset : list, optional
        Subset of subject IDs to load. If None, loads pilot subject "01".
    """

    def __init__(
        self,
        data_dir=None,
        max_iterations=500,
        derivative="enigma",
        subject_subset=None,
    ):
        if subject_subset is None:
            subject_subset = ["01"]

        # Caller passes list of struct names (e.g. ["L_Hipp", "R_Hipp", "BrStem"])
        # We'll partially initialize the loader to receive structs dynamically
        mesh_loader_kwargs = dict(
            derivative=derivative,
            subject_subset=subject_subset,
            as_mesh=True,
        )
        if data_dir is not None:
            mesh_loader_kwargs["data_dir"] = data_dir

        super().__init__(
            steps=[
                # Input: list of struct names like ["L_Hipp", "R_Hipp", ...]
                ppdict.HashWithIncoming(
                    step=Map(
                        step=Pipeline(
                            steps=[
                                PartiallyInitializedStep(
                                    Step=lambda struct_subset, **kwargs: MaternalMeshLoader(
                                        struct_subset=struct_subset, **kwargs
                                    ),
                                    pass_data=False,
                                    _struct_subset=lambda name: [name],
                                    **mesh_loader_kwargs,
                                ),
                                # Extract unique key to flatten nested structure (subject/session)
                                ppdict.ExtractUniqueKey(nested=True),
                            ]
                        )
                    )
                ),
                ppdict.DictMap(
                    step=PartiallyInitializedStep(
                        Step=lambda target, max_iterations: ppdict.DictMap(
                            PvAlign(target=target, max_iterations=max_iterations)
                            + TrimeshFromPvMesh()
                        ),
                        _target=lambda meshes: meshes[list(meshes.keys())[0]],
                        max_iterations=max_iterations,
                    )
                ),
            ]
        )
