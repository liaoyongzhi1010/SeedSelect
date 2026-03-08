import os
import numpy as np


def _set_egl():
    if 'PYOPENGL_PLATFORM' not in os.environ:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'


def render_mesh(mesh, camera_pose, width=320, height=320, yfov=np.deg2rad(60.0)):
    """Render RGB and depth using pyrender.

    Returns:
        color: HxWx3 uint8
        depth: HxW float32 (meters)
    """
    _set_egl()
    try:
        import pyrender
    except Exception as e:
        raise RuntimeError('pyrender not available; install requirements and ensure EGL is configured') from e

    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.5, 0.5, 0.5])
    if hasattr(mesh, 'visual'):
        pm = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    else:
        pm = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pm)

    camera = pyrender.PerspectiveCamera(yfov=yfov)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    color = color[:, :, :3]
    return color, depth
