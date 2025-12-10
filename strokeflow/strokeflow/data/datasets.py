import os
import random
from glob import glob
from typing import Tuple, Optional, List

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate


class StrokeFlowDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: Tuple[int, int, int] = (128, 128, 128),
        split_ratio: float = 0.8,
        seed: int = 42,
        augment: bool = True,
        use_mask: bool = True,
        img_folder: str = "images",
        density_folder: str = "density_gt",
        mask_folder: str = "mask_gt",
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.use_mask = use_mask

        self.img_dir = os.path.join(root, img_folder)
        self.dens_dir = os.path.join(root, density_folder)
        self.mask_dir = os.path.join(root, mask_folder)

        all_files = sorted(glob(os.path.join(self.img_dir, "*.nii*")))
        case_ids = [self._get_case_id(f) for f in all_files]
        
        rng = np.random.RandomState(seed)
        indices = np.arange(len(case_ids))
        rng.shuffle(indices)
        
        split_point = int(len(indices) * split_ratio)
        if split == "train":
            self.indices = indices[:split_point]
        else:
            self.indices = indices[split_point:]
            
        self.case_ids = [case_ids[i] for i in self.indices]

    @staticmethod
    def _get_case_id(path):
        filename = os.path.basename(path)
        if filename.endswith(".nii.gz"):
            return filename[:-7]
        elif filename.endswith(".nii"):
            return filename[:-4]
        return filename

    def _load_nifti(self, path):
        if not os.path.exists(path):
            alt_path = path.replace(".nii.gz", ".nii")
            if os.path.exists(alt_path):
                path = alt_path
            else:
                raise FileNotFoundError(f"File not found: {path}")
                
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        
        if data.ndim == 3:
            data = data[np.newaxis, ...] 
        elif data.ndim == 4:
            data = np.transpose(data, (3, 0, 1, 2))
            
        return data

    def _pad_or_crop(self, data, value=0.0):
        c, d, h, w = data.shape
        td, th, tw = self.img_size
        
        pad_d = max(0, td - d)
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pad_d1 = pad_d // 2
            pad_d2 = pad_d - pad_d1
            pad_h1 = pad_h // 2
            pad_h2 = pad_h - pad_h1
            pad_w1 = pad_w // 2
            pad_w2 = pad_w - pad_w1
            data = np.pad(
                data, 
                ((0, 0), (pad_d1, pad_d2), (pad_h1, pad_h2), (pad_w1, pad_w2)), 
                mode='constant', 
                constant_values=value
            )
            
        c, d, h, w = data.shape
        start_d = (d - td) // 2
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        
        return data[:, start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]

    def _normalize(self, data):
        mask = data > 0
        if np.any(mask):
            mean = np.mean(data[mask])
            std = np.std(data[mask]) + 1e-8
            data = (data - mean) / std
        else:
            data = (data - data.mean()) / (data.std() + 1e-8)
        return data

    def _augment(self, img, dens, mask):
        # 1. Random Flip
        if random.random() < 0.5:
            axis = random.choice([1, 2, 3]) 
            img = np.flip(img, axis=axis).copy()
            dens = np.flip(dens, axis=axis).copy()
            if mask is not None:
                mask = np.flip(mask, axis=axis).copy()

        # 2. Random 90 Rotations
        if random.random() < 0.5:
            k = random.randint(1, 3)
            axes = random.choice([(1, 2), (1, 3), (2, 3)])
            img = np.rot90(img, k=k, axes=axes).copy()
            dens = np.rot90(dens, k=k, axes=axes).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=axes).copy()
                
        # 3. Gamma Correction (Intensity only)
        if random.random() < 0.3:
            gamma = np.random.uniform(0.7, 1.5)
            min_v, max_v = img.min(), img.max()
            if max_v - min_v > 1e-6:
                img_norm = (img - min_v) / (max_v - min_v)
                img = img_norm ** gamma
                img = img * (max_v - min_v) + min_v
                
        return img, dens, mask

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        
        img_path = os.path.join(self.img_dir, f"{cid}.nii.gz")
        dens_path = os.path.join(self.dens_dir, f"{cid}.nii.gz")
        
        img = self._load_nifti(img_path)
        dens = self._load_nifti(dens_path)
        
        mask = None
        if self.use_mask:
            mask_path = os.path.join(self.mask_dir, f"{cid}.nii.gz")
            if os.path.exists(mask_path) or os.path.exists(mask_path.replace(".nii.gz", ".nii")):
                mask = self._load_nifti(mask_path)
            else:
                mask = np.zeros_like(dens)

        # Normalization
        for c in range(img.shape[0]):
            img[c] = self._normalize(img[c])

        # Padding/Cropping to fixed size
        img = self._pad_or_crop(img, value=img.min())
        dens = self._pad_or_crop(dens, value=0)
        if mask is not None:
            mask = self._pad_or_crop(mask, value=0)

        # Augmentation
        if self.augment:
            img, dens, mask = self._augment(img, dens, mask)

        # To Tensor
        img_t = torch.from_numpy(img).float()
        dens_t = torch.from_numpy(dens).float()
        
        # Ensure Density is [0, 1]
        dens_t = torch.clamp(dens_t, 0.0, 1.0)
        
        sample = {
            "id": cid,
            "image": img_t,
            "density_gt": dens_t,
            "mask": None
        }
        
        if mask is not None:
            mask_t = torch.from_numpy(mask).float()
            mask_t = (mask_t > 0.5).float()
            sample["mask"] = mask_t
            
        return sample