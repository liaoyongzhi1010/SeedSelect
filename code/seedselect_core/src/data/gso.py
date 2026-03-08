import json
import os
from typing import List


class GSOIndex:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.models_root = os.path.join(self.root, 'models_orig')
        self.split_path = os.path.join(self.root, 'train_test_split.json')
        self.valid_meshes_path = os.path.join(self.root, 'valid_meshes.json')

    def list_all_objects(self) -> List[str]:
        if os.path.exists(self.valid_meshes_path):
            with open(self.valid_meshes_path, 'r') as f:
                data = json.load(f)
            return data
        return sorted([d for d in os.listdir(self.models_root) if os.path.isdir(os.path.join(self.models_root, d))])

    def list_split(self, split='test') -> List[str]:
        if not os.path.exists(self.split_path):
            return self.list_all_objects()
        with open(self.split_path, 'r') as f:
            data = json.load(f)
        return data.get(split, [])

    def mesh_path(self, obj_id: str) -> str:
        return os.path.join(self.models_root, obj_id, 'meshes', 'model.obj')

    def thumbnails(self, obj_id: str) -> List[str]:
        tdir = os.path.join(self.models_root, obj_id, 'thumbnails')
        if not os.path.exists(tdir):
            return []
        imgs = [os.path.join(tdir, f) for f in os.listdir(tdir) if f.lower().endswith(('.jpg', '.png'))]
        return sorted(imgs)
