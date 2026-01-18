#!/usr/bin/env python3
"""
Pre-compute mesh predictions for all gestational weeks and export to JSON.

This script loads the trained models from the HerBrain pregnancy app and
generates mesh predictions for weeks 0-40, saving them in a format suitable
for the Astro frontend.

Usage:
    python precompute_meshes.py --data-dir /path/to/data --output ../public/data/prerendered_meshes.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to import herbrain modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from polpo.preprocessing import ListSqueeze
from polpo.preprocessing.learning import DictsToXY, NestedDictsToXY
from polpo.models import DictMeshColorizer, MeshColorizer
from polpo.sklearn.compose import PostTransformingEstimator

from herbrain.pregnancy.data import (
    HormonesCsvLoader,
    MaternalRegisteredMeshesLoader,
    MultipleMaternalMeshesLoader,
)
from herbrain.pregnancy.models import MeshPCR


def get_affine_transform():
    """Affine transformation to center the subcortical structures."""
    return np.array([
        [1.0, 0.0, 0.0, -23.0],
        [0.0, 1.0, 0.0, -9.0],
        [0.0, 0.0, 1.0, 27.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def train_week_mesh_model(data_dir: str, data_type: str = "multiple"):
    """Train the gestational week mesh model."""
    pregnancy_data_dir = os.path.join(data_dir, "pregnancy")
    maternal_data_dir = os.path.join(data_dir, "maternal")
    
    # Load hormone data for gestational weeks
    hormones_df = HormonesCsvLoader(data_dir=pregnancy_data_dir)()
    hormones_gest_week = ppd.ColumnToDict("gestWeek")(hormones_df)
    
    # Load registered meshes
    affine_transform = get_affine_transform()
    
    if data_type == "multiple":
        structs = [
            "L_Thal", "R_Thal",
            "L_Caud", "R_Caud",
            "L_Puta", "R_Puta",
            "L_Pall", "R_Pall",
            "L_Hipp", "R_Hipp",
            "L_Amyg", "R_Amyg",
            "L_Accu", "R_Accu",
        ]
        n_structs = len(structs)
        registered_meshes = MultipleMaternalMeshesLoader(
            data_dir=maternal_data_dir, max_iterations=500
        )(structs)
        dicts_to_xy = NestedDictsToXY()
        n_pipes = n_structs
    else:
        registered_meshes = MaternalRegisteredMeshesLoader(
            data_dir=maternal_data_dir, max_iterations=500
        )()
        dicts_to_xy = DictsToXY()
        n_pipes = None
        n_structs = 1
    
    # Create and train the model
    week_mesh_model = MeshPCR(
        model=None, affine_transform=affine_transform, n_pipes=n_pipes
    )
    
    # Add colorizer
    Colorizer = DictMeshColorizer if data_type == "multiple" else MeshColorizer
    week_colorizer = Colorizer(x_ref=np.asarray(0.5), delta_lim=np.asarray(15.0))
    week_mesh_model = PostTransformingEstimator(week_mesh_model, week_colorizer)
    
    # Fit the model
    X, y = dicts_to_xy([hormones_gest_week, registered_meshes])
    week_mesh_model.fit(X, y)
    
    # Set up post-processing for multiple structures
    postproc_pred = None
    if data_type == "multiple":
        postproc_pred = ppdict.DictMap(step=ListSqueeze()) + ppdict.DictToValuesList()
    
    return week_mesh_model, postproc_pred, n_structs, structs if data_type == "multiple" else ["L_Hipp"]


def mesh_to_dict(mesh, struct_name: str) -> dict:
    """Convert a trimesh object to a JSON-serializable dictionary."""
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    
    # Extract vertex colors if available
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3].astype(float) / 255.0
        colors = colors.tolist()
    else:
        # Default beige color
        colors = [[0.9, 0.8, 0.7]] * len(vertices)
    
    return {
        "name": struct_name,
        "vertices": vertices,
        "faces": faces,
        "colors": colors,
    }


def generate_predictions(
    week_mesh_model,
    postproc_pred,
    struct_names: list,
    weeks: list = None
) -> dict:
    """Generate mesh predictions for all specified weeks."""
    if weeks is None:
        weeks = list(range(0, 41, 5))  # Every 5 weeks: 0, 5, 10, ..., 40
    
    results = {"weeks": {}, "metadata": {
        "colorScale": {
            "shrink": [1.0, 0.0, 0.0],  # Red
            "grow": [0.0, 0.0, 1.0],     # Blue
            "unchanged": [0.9, 0.8, 0.7], # Beige
        }
    }}
    
    for week in weeks:
        print(f"Generating predictions for week {week}...")
        try:
            # Predict mesh for this week
            result = week_mesh_model.predict(np.array([[week]]))
            
            if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                mesh_data = result[0]
            else:
                mesh_data = result
            
            if postproc_pred is not None:
                mesh_data = postproc_pred(mesh_data)
            
            # Convert meshes to JSON format
            if isinstance(mesh_data, list):
                structures = [
                    mesh_to_dict(mesh, struct_names[i])
                    for i, mesh in enumerate(mesh_data)
                ]
            else:
                structures = [mesh_to_dict(mesh_data, struct_names[0])]
            
            results["weeks"][str(week)] = {"structures": structures}
            
        except Exception as e:
            print(f"Warning: Could not generate prediction for week {week}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute mesh predictions for gestational weeks"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("HERBRAIN_DATA_DIR", "~/.herbrain/data/"),
        help="Directory containing herbrain data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../public/data/prerendered_meshes.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="multiple",
        choices=["single", "multiple"],
        help="Data type: 'single' for one structure, 'multiple' for all subcortical",
    )
    parser.add_argument(
        "--weeks",
        type=str,
        default="0,5,10,15,20,25,30,35,40",
        help="Comma-separated list of weeks to pre-compute",
    )
    
    args = parser.parse_args()
    
    # Expand user path
    data_dir = os.path.expanduser(args.data_dir)
    
    # Parse weeks
    weeks = [int(w.strip()) for w in args.weeks.split(",")]
    
    print(f"Loading models from {data_dir}...")
    week_mesh_model, postproc_pred, n_structs, struct_names = train_week_mesh_model(
        data_dir, args.data_type
    )
    
    print(f"Generating predictions for {len(weeks)} weeks...")
    results = generate_predictions(
        week_mesh_model, postproc_pred, struct_names, weeks
    )
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to JSON
    print(f"Writing results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, separators=(",", ":"))
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done! Generated {len(results['weeks'])} weeks of data ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
