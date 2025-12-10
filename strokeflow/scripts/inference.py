import argparse
import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import Tuple, List, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from strokeflow.config import StrokeFlowConfig
from strokeflow.models import StrokeFlowUNet3D

def _get_gaussian(patch_size, sigma_scale=1. / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = np.zeros_like(tmp)
    gaussian_importance_map[tuple(center_coords)] = 1
    
    for i in range(len(patch_size)):
        d = np.arange(patch_size[i]) - center_coords[i]
        gaussian_importance_map = np.exp(-((d ** 2) / (2 * sigmas[i] ** 2))) * gaussian_importance_map
        
    return gaussian_importance_map / np.max(gaussian_importance_map)

class SlidingWindowInferer:
    def __init__(self, model, patch_size=(128, 128, 128), overlap=0.5, batch_size=1, device='cuda'):
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.device = device
        self.gaussian_map = _get_gaussian(patch_size)
        self.gaussian_map = torch.from_numpy(self.gaussian_map).float().to(device)

    def _compute_steps_for_sliding_window(self, image_size, patch_size, overlap):
        steps = []
        for dim in range(len(image_size)):
            max_p = image_size[dim]
            min_p = patch_size[dim]
            
            if max_p <= min_p:
                steps.append([0])
                continue
                
            stride = int(min_p * (1 - overlap))
            num_steps = math.ceil((max_p - min_p) / stride) + 1
            
            dim_steps = []
            for i in range(num_steps):
                step = i * stride
                if step + min_p > max_p:
                    step = max_p - min_p
                if step not in dim_steps:
                    dim_steps.append(step)
            steps.append(dim_steps)
        return steps

    def __call__(self, input_volume):
        self.model.eval()
        with torch.no_grad():
            if input_volume.ndim == 3:
                input_volume = input_volume.unsqueeze(0)
            if input_volume.ndim == 4 and input_volume.shape[0] != 1:
                 input_volume = input_volume.unsqueeze(0)

            input_volume = input_volume.to(self.device)
            b, c, d, h, w = input_volume.shape
            image_size = (d, h, w)
            
            pred_density = torch.zeros((1, 1, d, h, w)).to(self.device)
            pred_flow = torch.zeros((1, 3, d, h, w)).to(self.device)
            count_map = torch.zeros((1, 1, d, h, w)).to(self.device)
            
            steps = self._compute_steps_for_sliding_window(image_size, self.patch_size, self.overlap)
            
            patches = []
            coords = []
            
            for z in steps[0]:
                for y in steps[1]:
                    for x in steps[2]:
                        s_z, s_y, s_x = z, y, x
                        e_z, e_y, e_x = z + self.patch_size[0], y + self.patch_size[1], x + self.patch_size[2]
                        
                        patch = input_volume[:, :, s_z:e_z, s_y:e_y, s_x:e_x]
                        
                        if patch.shape[2:] != self.patch_size:
                            pad_d = self.patch_size[0] - patch.shape[2]
                            pad_h = self.patch_size[1] - patch.shape[3]
                            pad_w = self.patch_size[2] - patch.shape[4]
                            patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d))
                        
                        patches.append(patch)
                        coords.append((s_z, s_y, s_x, e_z, e_y, e_x))

            num_patches = len(patches)
            for i in range(0, num_patches, self.batch_size):
                batch_patches = patches[i:i+self.batch_size]
                batch_coords = coords[i:i+self.batch_size]
                
                input_tensor = torch.cat(batch_patches, dim=0).to(self.device)
                
                res_density, res_flow = self.model(input_tensor)
                
                for j in range(len(batch_coords)):
                    sz, sy, sx, ez, ey, ex = batch_coords[j]
                    
                    p_dens = res_density[j:j+1]
                    p_flow = res_flow[j:j+1]
                    
                    valid_d = min(ez, d) - sz
                    valid_h = min(ey, h) - sy
                    valid_w = min(ex, w) - sx
                    
                    p_dens = p_dens[:, :, :valid_d, :valid_h, :valid_w]
                    p_flow = p_flow[:, :, :valid_d, :valid_h, :valid_w]
                    weight = self.gaussian_map[:valid_d, :valid_h, :valid_w]

                    pred_density[:, :, sz:sz+valid_d, sy:sy+valid_h, sx:sx+valid_w] += p_dens * weight
                    pred_flow[:, :, sz:sz+valid_d, sy:sy+valid_h, sx:sx+valid_w] += p_flow * weight
                    count_map[:, :, sz:sz+valid_d, sy:sy+valid_h, sx:sx+valid_w] += weight

            count_map = torch.clamp(count_map, min=1e-6)
            pred_density /= count_map
            pred_flow /= count_map
            
            return pred_density, pred_flow

def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    elif data.ndim == 4:
        data = np.transpose(data, (3, 2, 1, 0)) 
        if data.shape[0] > 2:
            data = data[:2, ...]
            
    return data, affine

def preprocess_volume(data):
    mean = np.mean(data)
    std = np.std(data) + 1e-8
    data = (data - mean) / std
    return torch.from_numpy(data).float()

def save_prediction(data, affine, output_path, is_vector=False):
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    
    if is_vector:
        data = np.transpose(data, (1, 2, 3, 0))
    else:
        data = data.squeeze()
        
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = StrokeFlowConfig.load_yaml(args.config)
    
    model = StrokeFlowUNet3D(in_channels=cfg.in_channels, base_filters=cfg.base_filters)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if args.input_path.endswith('.nii') or args.input_path.endswith('.nii.gz'):
        file_list = [args.input_path]
    else:
        file_list = sorted(glob(os.path.join(args.input_path, "*.nii.gz")))
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "density"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "flow"), exist_ok=True)
    
    inferer = SlidingWindowInferer(
        model=model,
        patch_size=cfg.img_size,
        overlap=args.overlap,
        batch_size=1,
        device=device
    )

    for fpath in file_list:
        filename = os.path.basename(fpath)
        case_id = filename.split('.')[0]
        
        raw_data, affine = load_nifti_file(fpath)
        input_tensor = preprocess_volume(raw_data)
        
        density_pred, flow_pred = inferer(input_tensor)
        
        mask_pred = (density_pred > 0.5).float()
        
        save_prediction(density_pred, affine, os.path.join(args.output_dir, "density", filename))
        save_prediction(mask_pred, affine, os.path.join(args.output_dir, "mask", filename))
        save_prediction(flow_pred, affine, os.path.join(args.output_dir, "flow", filename), is_vector=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--overlap", type=float, default=0.5)
    
    args = parser.parse_args()
    run_inference(args)