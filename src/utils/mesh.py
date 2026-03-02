import numpy as np
import trimesh


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def normalize_mesh(mesh, mode='unit_cube'):
    """Normalize mesh to fit within [-1,1]^3 by default."""
    vertices = mesh.vertices.copy()
    if mode == 'unit_cube':
        min_v = vertices.min(axis=0)
        max_v = vertices.max(axis=0)
        center = (min_v + max_v) / 2.0
        scale = (max_v - min_v).max() / 2.0
        scale = max(scale, 1e-8)
        vertices = (vertices - center) / scale
    elif mode == 'unit_sphere':
        center = vertices.mean(axis=0)
        vertices = vertices - center
        scale = np.linalg.norm(vertices, axis=1).max()
        scale = max(scale, 1e-8)
        vertices = vertices / scale
    mesh = mesh.copy()
    mesh.vertices = vertices
    return mesh
