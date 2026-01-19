import os
import glob
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np
import open3d as o3d

import pointops
from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose

# def load_xyz_from_pcd(pcd_path: str) -> np.ndarray:
#     """
#     Minimal ASCII .pcd reader for XYZ (and optionally normals if present).
#     Assumes ASCII PCD. If your PCD is binary, you must convert or use a proper reader.
#     Returns Nx3 float32.
#     """
#     header = []
#     with open(pcd_path, "r") as f:
#         while True:
#             line = f.readline()
#             if not line:
#                 raise ValueError(f"Invalid PCD (no DATA line): {pcd_path}")
#             line = line.strip()
#             header.append(line)
#             if line.upper().startswith("DATA"):
#                 data_type = line.split()[1].lower()
#                 if data_type != "ascii":
#                     raise ValueError(f"Only ASCII PCD supported in this loader: {pcd_path} (DATA {data_type})")
#                 break

#         # Find FIELDS to know which columns exist
#         fields = None
#         for h in header:
#             if h.upper().startswith("FIELDS"):
#                 fields = h.split()[1:]
#                 break
#         if fields is None:
#             raise ValueError(f"PCD missing FIELDS line: {pcd_path}")

#         # Read points
#         pts = []
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             vals = line.split()
#             pts.append([float(v) for v in vals])

#     arr = np.asarray(pts, dtype=np.float32)

#     # Map field names to indices
#     field_to_idx = {name: i for i, name in enumerate(fields)}
#     for k in ("x", "y", "z"):
#         if k not in field_to_idx:
#             raise ValueError(f"PCD missing {k} field: {pcd_path} (fields={fields})")

#     xyz = arr[:, [field_to_idx["x"], field_to_idx["y"], field_to_idx["z"]]]
#     return xyz
def load_xyz_from_pcd_open3d(pcd_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)  # Nx3
    if xyz.size == 0:
        raise ValueError(f"Empty point cloud: {pcd_path}")
    return xyz

@DATASETS.register_module()
class LidarDataset(Dataset):
    def __init__(
        self,
        split="train",                 # "train" or "test"
        data_root="data/lidar",
        class_names=None,              # optional; if None, inferred from subfolders
        transform=None,
        num_points=8192,
        uniform_sampling=True,
        test_mode=False,
        test_cfg=None,
        loop=1,
        split_ratio=0.8,               # 80/20 per class
        seed=0,                        # for reproducible splits
        save_record=True,
    ):
        super().__init__()
        self.data_root = data_root

        if class_names is None:
            class_names = sorted(
                d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            )
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.class_names = class_names

        self.split = split
        self.split_ratio = float(split_ratio)
        self.seed = int(seed)

        self.num_point = num_points
        self.uniform_sampling = uniform_sampling
        self.transform = Compose(transform or [])

        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()

        logger = get_root_logger()
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Totally {len(self.data_list)} x {self.loop} samples in {split} set.")

        # Optional record cache (saves loaded+sampled data)
        record_name = f"lidar_{self.split}_seed{self.seed}_r{int(self.split_ratio*100)}"
        if num_points is not None:
            record_name += f"_{num_points}points"
            if uniform_sampling:
                record_name += "_uniform"
        record_path = os.path.join(self.data_root, f"{record_name}.pth")

        if os.path.isfile(record_path):
            logger.info(f"Loading record: {record_name} ...")
            self.data = torch.load(record_path, weights_only=False)
        else:
            logger.info(f"Preparing record: {record_name} ...")
            self.data = {}
            for idx in range(len(self.data_list)):
                self.data[self.get_data_name(idx)] = self.get_data(idx)
            if save_record:
                torch.save(self.data, record_path)

    def get_data_list(self):
        # Build per-class list of files, then split 80/20 per class (reproducible)
        items = []
        rng = np.random.RandomState(self.seed)

        for cls in self.class_names:
            cls_dir = os.path.join(self.data_root, cls)
            files = sorted(glob.glob(os.path.join(cls_dir, "*.pcd")))

            if len(files) == 0:
                raise FileNotFoundError(f"No .pcd files found in {cls_dir}")

            idxs = np.arange(len(files))
            rng.shuffle(idxs)

            cut = int(round(self.split_ratio * len(files)))
            if self.split == "train":
                chosen = idxs[:cut]
            elif self.split in ("test", "val"):
                chosen = idxs[cut:]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            for i in chosen:
                items.append((cls, files[i]))

        # Shuffle overall list for training (optional)
        if self.split == "train":
            rng.shuffle(items)
        return items

    def get_data_name(self, idx):
        cls, path = self.data_list[idx % len(self.data_list)]
        # unique key for caching
        rel = os.path.relpath(path, self.data_root)
        return f"{cls}::{rel}"




    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        cls, pcd_path = self.data_list[data_idx]

        # Cache key (optional, wenn du record nutzt)
        key = self.get_data_name(data_idx)
        if key in self.data:
            return copy.deepcopy(self.data[key])

        xyz = load_xyz_from_pcd_open3d(pcd_path)  # Nx3

        if self.num_point is not None:
            if self.uniform_sampling:
                assert torch.cuda.is_available(), "uniform_sampling=True requires CUDA pointops"
                with torch.no_grad():
                    pts = torch.from_numpy(xyz).float().cuda()
                    mask = pointops.farthest_point_sampling(
                        pts,
                        torch.tensor([len(pts)]).long().cuda(),
                        torch.tensor([self.num_point]).long().cuda(),
                    )
                sel = mask.squeeze(0).cpu().numpy()
                xyz = xyz[sel]
            else:
                xyz = xyz[: self.num_point]

        category = np.array([self.class_to_idx[cls]], dtype=np.int64)

        # Falls dein Modell/Transforms "normal" erwarten:
        normal = np.zeros_like(xyz, dtype=np.float32)

        out = dict(coord=xyz.astype(np.float32), normal=normal, category=category)

        # falls du record-caching benutzt:
        self.data[key] = copy.deepcopy(out)
        return out


    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        return self.transform(data_dict)

    def prepare_test_data(self, idx):
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        category = data_dict.pop("category")
        data_dict = self.transform(data_dict)

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        for i in range(len(data_dict_list)):
            data_dict_list[i] = self.post_transform(data_dict_list[i])

        return dict(
            voting_list=data_dict_list,
            category=category,
            name=self.get_data_name(idx),
        )
