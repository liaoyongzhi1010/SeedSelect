import os
from typing import List, Tuple


class OmniObject3DIndex:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.render_root = os.path.join(self.root, 'blender_renders')
        self.mesh_root = os.path.join(self.root, 'meshes')

    def list_objects(self) -> List[str]:
        if not os.path.exists(self.render_root):
            return []
        return sorted([d for d in os.listdir(self.render_root) if os.path.isdir(os.path.join(self.render_root, d))])

    def category_from_id(self, obj_id: str) -> str:
        # object ids are like 'battery_001'
        if '_' in obj_id:
            return obj_id.split('_')[0]
        return obj_id

    def list_by_category(self) -> List[Tuple[str, List[str]]]:
        cats = {}
        for obj_id in self.list_objects():
            cat = self.category_from_id(obj_id)
            cats.setdefault(cat, []).append(obj_id)
        return [(k, sorted(v)) for k, v in sorted(cats.items())]

    def mesh_path(self, obj_id: str) -> str:
        return os.path.join(self.mesh_root, f'{obj_id}.obj')

    def render_dir(self, obj_id: str) -> str:
        return os.path.join(self.render_root, obj_id, 'render')

    def images_dir(self, obj_id: str) -> str:
        return os.path.join(self.render_dir(obj_id), 'images')

    def depths_dir(self, obj_id: str) -> str:
        return os.path.join(self.render_dir(obj_id), 'depths')

    def transforms_path(self, obj_id: str) -> str:
        return os.path.join(self.render_dir(obj_id), 'transforms.json')
