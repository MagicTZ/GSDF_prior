#!/usr/bin/env python3
"""
Mesh Coordinate Transformation Tool for GSDF

This script applies inverse transformation to convert mesh from normalized 
coordinate system back to the original coordinate system.

Usage:
    # Transform a single mesh using saved transformation parameters
    python transform_mesh.py --input mesh_normalized.ply --transform transform_params_sdf.json --output mesh_original.ply
    
    # Transform with manually specified parameters
    python transform_mesh.py --input mesh_normalized.ply --center 0.5 0.5 0.5 --scale 2.0 --output mesh_original.ply
    
    # Batch transform all meshes in a directory
    python transform_mesh.py --input_dir ./exp/scene/trial/save/ --transform ./data/scene/transform_params_sdf.json
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

try:
    import trimesh
except ImportError:
    print("Error: trimesh is not installed. Please install it using:")
    print("  pip install trimesh")
    sys.exit(1)


def load_transformation_params(transform_file):
    """Load transformation parameters from JSON file."""
    with open(transform_file, 'r') as f:
        params = json.load(f)
    
    center = np.array(params['center'], dtype=np.float32)
    scale = float(params['scale'])
    
    print(f"Loaded transformation parameters:")
    print(f"  Center: {center}")
    print(f"  Scale: {scale}")
    
    return center, scale


def apply_inverse_transform(vertices, center, scale):
    """
    Apply inverse transformation to convert normalized coordinates back to original.
    
    Forward transformation (normalization):
        vertices_norm = (vertices - center) / scale
    
    Inverse transformation:
        vertices_original = vertices_norm * scale + center
    
    Args:
        vertices: numpy array of shape (N, 3)
        center: numpy array of shape (3,) - the center that was subtracted
        scale: float - the scale factor that was applied
    
    Returns:
        transformed vertices in original coordinate system
    """
    # Apply inverse scale
    vertices_scaled = vertices * scale
    
    # Apply inverse translation (add back center)
    vertices_original = vertices_scaled + center.reshape(1, 3)
    
    return vertices_original


def transform_mesh(input_path, output_path, center, scale):
    """Transform a single mesh file."""
    print(f"\nTransforming mesh: {input_path}")
    
    # Load mesh
    mesh = trimesh.load(input_path, process=False)
    
    # Get original bounds
    original_bounds = mesh.bounds
    print(f"  Original bounds: {original_bounds[0]} to {original_bounds[1]}")
    
    # Apply inverse transformation
    mesh.vertices = apply_inverse_transform(mesh.vertices, center, scale)
    
    # Get transformed bounds
    transformed_bounds = mesh.bounds
    print(f"  Transformed bounds: {transformed_bounds[0]} to {transformed_bounds[1]}")
    
    # Save transformed mesh
    mesh.export(output_path)
    print(f"  Saved to: {output_path}")
    
    return True


def batch_transform_meshes(input_dir, transform_file, output_dir=None, suffix='_original'):
    """Transform all mesh files in a directory."""
    input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load transformation parameters
    center, scale = load_transformation_params(transform_file)
    
    # Find all mesh files
    mesh_extensions = ['.ply', '.obj', '.stl', '.off']
    mesh_files = []
    for ext in mesh_extensions:
        mesh_files.extend(list(input_dir.glob(f'*{ext}')))
    
    if not mesh_files:
        print(f"No mesh files found in {input_dir}")
        return
    
    print(f"\nFound {len(mesh_files)} mesh file(s) to transform")
    
    # Transform each mesh
    for mesh_file in mesh_files:
        # Generate output filename
        output_name = mesh_file.stem + suffix + mesh_file.suffix
        output_path = output_dir / output_name
        
        try:
            transform_mesh(str(mesh_file), str(output_path), center, scale)
        except Exception as e:
            print(f"  Error transforming {mesh_file}: {e}")
            continue
    
    print(f"\nBatch transformation completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Transform mesh from normalized coordinates back to original coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Input mesh file path')
    input_group.add_argument('--input_dir', type=str, help='Input directory containing mesh files')
    
    # Transformation parameters
    transform_group = parser.add_mutually_exclusive_group(required=True)
    transform_group.add_argument('--transform', type=str, 
                                  help='Path to transformation parameters JSON file (transform_params_sdf.json or transform_params_gs.json)')
    transform_group.add_argument('--manual', action='store_true',
                                  help='Manually specify transformation parameters')
    
    # Manual parameters (used with --manual)
    parser.add_argument('--center', type=float, nargs=3, 
                       help='Center coordinates (x y z) used in normalization')
    parser.add_argument('--scale', type=float, 
                       help='Scale factor used in normalization')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output mesh file path (for single file mode)')
    parser.add_argument('--output_dir', type=str, help='Output directory (for batch mode)')
    parser.add_argument('--suffix', type=str, default='_original',
                       help='Suffix to add to output filenames in batch mode (default: _original)')
    
    args = parser.parse_args()
    
    # Get transformation parameters
    if args.transform:
        if not os.path.exists(args.transform):
            print(f"Error: Transformation file not found: {args.transform}")
            sys.exit(1)
        center, scale = load_transformation_params(args.transform)
    elif args.manual:
        if args.center is None or args.scale is None:
            print("Error: --center and --scale must be specified when using --manual")
            sys.exit(1)
        center = np.array(args.center, dtype=np.float32)
        scale = args.scale
        print(f"Using manual transformation parameters:")
        print(f"  Center: {center}")
        print(f"  Scale: {scale}")
    
    # Single file mode
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_path = input_path.parent / (input_path.stem + '_original' + input_path.suffix)
        
        transform_mesh(args.input, output_path, center, scale)
        print(f"\n✓ Transformation completed successfully!")
    
    # Batch directory mode
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)
        
        batch_transform_meshes(args.input_dir, args.transform, args.output_dir, args.suffix)
        print(f"\n✓ Batch transformation completed successfully!")


if __name__ == '__main__':
    main()
