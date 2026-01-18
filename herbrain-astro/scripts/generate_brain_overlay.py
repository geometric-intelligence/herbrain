#!/usr/bin/env python3
"""
Generate the brain overlay mesh data and add it to prerendered_meshes.json.

The brain overlay is a static, semi-transparent mesh of the whole brain template
that provides context for the subcortical structures visualization.

Usage:
    python generate_brain_overlay.py --output ../public/data/prerendered_meshes.json
    
    # Or with MRI data (if available):
    python generate_brain_overlay.py --data-dir /path/to/data --use-mri
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to import herbrain modules (if available)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_brain_ellipsoid_mesh(center, radii, resolution=50):
    """
    Create a brain-like ellipsoid mesh.
    
    The brain is approximated as an ellipsoid with the following adjustments:
    - Slightly flattened at the front (prefrontal area)
    - Rounded at the back (occipital area)
    
    Parameters
    ----------
    center : tuple
        (x, y, z) center of the brain
    radii : tuple
        (rx, ry, rz) radii in x, y, z directions
    resolution : int
        Number of subdivisions for mesh resolution
        
    Returns
    -------
    vertices : np.ndarray
        Array of vertex positions
    faces : np.ndarray
        Array of face indices
    """
    # Create parametric surface
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2)
    u, v = np.meshgrid(u, v)
    
    # Basic ellipsoid
    x = radii[0] * np.sin(v) * np.cos(u) + center[0]
    y = radii[1] * np.sin(v) * np.sin(u) + center[1]
    z = radii[2] * np.cos(v) + center[2]
    
    # Flatten the arrays
    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # Create faces (triangles)
    faces = []
    n_u, n_v = resolution, resolution // 2
    for i in range(n_v - 1):
        for j in range(n_u - 1):
            # Current vertex indices
            v0 = i * n_u + j
            v1 = i * n_u + (j + 1)
            v2 = (i + 1) * n_u + j
            v3 = (i + 1) * n_u + (j + 1)
            
            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return vertices, np.array(faces)


def create_brain_mesh_from_subcortical_bounds(prerendered_path: str, padding_factor=1.8):
    """
    Create a brain overlay mesh sized to encompass subcortical structures.
    
    Reads the prerendered meshes to determine the bounding box of subcortical
    structures, then creates a brain-shaped ellipsoid that encompasses them.
    
    Parameters
    ----------
    prerendered_path : str
        Path to prerendered_meshes.json
    padding_factor : float
        How much larger the brain should be compared to subcortical bounds
        
    Returns
    -------
    vertices : np.ndarray
        Brain mesh vertices
    faces : np.ndarray
        Brain mesh faces
    """
    import base64
    
    def decode_plotly_array(data):
        """Decode Plotly binary array."""
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, dict) and 'bdata' in data:
            dtype_map = {
                'i1': np.int8, 'i2': np.int16, 'i4': np.int32, 'i8': np.int64,
                'u1': np.uint8, 'u2': np.uint16, 'u4': np.uint32, 'u8': np.uint64,
                'f4': np.float32, 'f8': np.float64,
            }
            dtype = dtype_map.get(data.get('dtype', 'f8'), np.float64)
            decoded = base64.b64decode(data['bdata'])
            return np.frombuffer(decoded, dtype=dtype)
        return np.array(data) if data is not None else np.array([])
    
    # Load prerendered meshes to get bounds
    with open(prerendered_path, 'r') as f:
        meshes = json.load(f)
    
    # Get coordinates from first available week
    first_week = list(meshes.keys())[0]
    traces = meshes[first_week].get('data', [])
    
    all_x, all_y, all_z = [], [], []
    for trace in traces:
        if trace.get('type') == 'mesh3d' and trace.get('name') != 'brain_overlay':
            x = decode_plotly_array(trace.get('x'))
            y = decode_plotly_array(trace.get('y'))
            z = decode_plotly_array(trace.get('z'))
            if len(x) > 0:
                all_x.extend(x.tolist())
                all_y.extend(y.tolist())
                all_z.extend(z.tolist())
    
    if not all_x:
        raise ValueError("No subcortical structure data found in prerendered meshes")
    
    # Calculate bounds and center
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)
    
    center = (
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        (z_min + z_max) / 2,
    )
    
    # Calculate radii with padding
    radii = (
        (x_max - x_min) / 2 * padding_factor,
        (y_max - y_min) / 2 * padding_factor,
        (z_max - z_min) / 2 * padding_factor,
    )
    
    print(f"Subcortical bounds: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], Z[{z_min:.1f}, {z_max:.1f}]")
    print(f"Brain center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"Brain radii: ({radii[0]:.1f}, {radii[1]:.1f}, {radii[2]:.1f})")
    
    return create_brain_ellipsoid_mesh(center, radii, resolution=60)


def vertices_faces_to_plotly_trace(vertices, faces, opacity=0.12, color='rgb(200, 200, 210)'):
    """Convert vertices and faces to a Plotly Mesh3d trace."""
    trace = {
        "type": "mesh3d",
        "x": vertices[:, 0].tolist(),
        "y": vertices[:, 1].tolist(),
        "z": vertices[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "color": color,
        "opacity": opacity,
        "name": "brain_overlay",
        "hoverinfo": "skip",
        "flatshading": False,
        "lighting": {
            "ambient": 0.7,
            "diffuse": 0.4,
            "specular": 0.05,
            "roughness": 0.95,
        },
        "lightposition": {
            "x": 100,
            "y": 200,
            "z": 100,
        },
    }
    
    return trace


def generate_brain_overlay_from_mri(data_dir: str):
    """Generate the brain overlay mesh from the template MRI image (requires data)."""
    try:
        from herbrain.pregnancy.data import TemplateImageLoader, NibImage2Mesh
    except ImportError:
        raise ImportError("herbrain package not available. Use --no-mri flag to generate approximate mesh.")
    
    pregnancy_data_dir = os.path.join(data_dir, "pregnancy")
    
    print(f"Loading template image from {pregnancy_data_dir}...")
    template_image = TemplateImageLoader(data_dir=pregnancy_data_dir)()
    
    print("Converting MRI image to mesh using marching cubes...")
    template_mesh = NibImage2Mesh()(template_image)
    
    print(f"Generated mesh with {len(template_mesh.vertices)} vertices and {len(template_mesh.faces)} faces")
    
    # Apply affine transform to match subcortical structures
    affine_transform = np.array([
        [1.0, 0.0, 0.0, -23.0],
        [0.0, 1.0, 0.0, -9.0],
        [0.0, 0.0, 1.0, 27.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    
    vertices = np.array(template_mesh.vertices)
    n_vertices = len(vertices)
    vertices_h = np.hstack([vertices, np.ones((n_vertices, 1))])
    vertices = (affine_transform @ vertices_h.T).T[:, :3]
    
    faces = np.array(template_mesh.faces)
    
    return vertices, faces


def generate_brain_overlay(prerendered_path: str, use_mri: bool = False, data_dir: str = None):
    """Generate the brain overlay mesh."""
    if use_mri and data_dir:
        print("Generating brain overlay from MRI data...")
        vertices, faces = generate_brain_overlay_from_mri(data_dir)
    else:
        print("Generating approximate brain overlay from subcortical bounds...")
        vertices, faces = create_brain_mesh_from_subcortical_bounds(prerendered_path)
    
    return vertices_faces_to_plotly_trace(vertices, faces)


def add_overlay_to_meshes(prerendered_path: str, overlay_trace: dict, output_path: str):
    """Add the brain overlay trace to all weeks in prerendered meshes."""
    print(f"Loading prerendered meshes from {prerendered_path}...")
    with open(prerendered_path, 'r') as f:
        meshes = json.load(f)
    
    print(f"Adding brain overlay to {len(meshes)} weeks...")
    
    for week in meshes:
        if 'data' in meshes[week]:
            # Remove existing overlay if present (to allow updates)
            meshes[week]['data'] = [
                t for t in meshes[week]['data'] 
                if t.get('name') != 'brain_overlay'
            ]
            # Add overlay as the first trace (renders behind subcortical structures)
            meshes[week]['data'].insert(0, overlay_trace)
    
    print(f"Writing updated meshes to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(meshes, f, separators=(',', ':'))
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! Output file size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate brain overlay mesh and add to prerendered meshes"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("HERBRAIN_DATA_DIR", "~/.herbrain/data/"),
        help="Directory containing herbrain data (only needed with --use-mri)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../public/data/prerendered_meshes.json",
        help="Input prerendered meshes JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (defaults to input path)",
    )
    parser.add_argument(
        "--use-mri",
        action="store_true",
        help="Generate brain mesh from MRI data (requires herbrain data)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.12,
        help="Opacity of the brain overlay (0.0 to 1.0)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="rgb(200, 200, 210)",
        help="Color of the brain overlay",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    
    if not input_path.is_absolute():
        input_path = (script_dir / input_path).resolve()
    if not output_path.is_absolute():
        output_path = (script_dir / output_path).resolve()
    
    # Generate the brain overlay
    data_dir = os.path.expanduser(args.data_dir) if args.use_mri else None
    overlay_trace = generate_brain_overlay(
        str(input_path), 
        use_mri=args.use_mri, 
        data_dir=data_dir
    )
    
    # Update opacity and color if specified
    overlay_trace['opacity'] = args.opacity
    overlay_trace['color'] = args.color
    
    # Add overlay to prerendered meshes
    add_overlay_to_meshes(str(input_path), overlay_trace, str(output_path))
    
    print("\nBrain overlay has been added to the prerendered meshes!")
    print("The overlay will be displayed as a semi-transparent brain in the 3D visualization.")


if __name__ == "__main__":
    main()
