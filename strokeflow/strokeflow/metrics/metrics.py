import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from math import exp, log10

def dice_coefficient(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return ((2.0 * inter + eps) / (union + eps)).item()

def iou(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return ((inter + eps) / (union + eps)).item()

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def psnr(pred, target, data_range=1.0, eps=1e-6):
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 100.0
    return 20 * log10(data_range) - 10 * log10(mse)

def soft_dice(pred, target, eps=1e-6):
    inter = torch.sum(pred * target)
    sum_ = torch.sum(pred * pred) + torch.sum(target * target)
    return (2.0 * inter + eps) / (sum_ + eps)

def soft_iou(pred, target, eps=1e-6):
    inter = torch.sum(pred * target)
    union = torch.sum(pred + target - pred * target)
    return (inter + eps) / (union + eps)

def density_concentration_ratio(pred_density, gt_mask, eps=1e-6):
    pred = pred_density.float()
    mask = gt_mask.float()
    inside = torch.sum(pred * mask)
    total = torch.sum(pred)
    return ((inside + eps) / (total + eps)).item()

def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window_3d(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def ssim_3d(pred, target, window_size=11, size_average=True):
    channel = pred.size(1)
    window = create_window_3d(window_size, channel)
    
    if pred.is_cuda:
        window = window.to(pred.device)
    window = window.type_as(pred)

    mu1 = F.conv3d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()

def hd95(pred, target, voxel_spacing=(1.0, 1.0, 1.0)):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    pred = pred.astype(bool)
    target = target.astype(bool)

    if not np.any(pred) and not np.any(target):
        return 0.0
    
    if not np.any(pred) or not np.any(target):
        return 373.13

    dt_target = distance_transform_edt(~target, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred, sampling=voxel_spacing)

    hd1 = np.percentile(dt_target[pred], 95)
    hd2 = np.percentile(dt_pred[target], 95)

    return float(max(hd1, hd2))

class StrokeFlowMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()

    def update(self, pred_density, gt_density, gt_mask=None):
        pred_density = pred_density.detach()
        gt_density = gt_density.detach()
        
        self.metrics["MAE"].append(mae(pred_density, gt_density))
        self.metrics["PSNR"].append(psnr(pred_density, gt_density))
        self.metrics["SSIM"].append(ssim_3d(pred_density, gt_density))
        self.metrics["SoftDice"].append(soft_dice(pred_density, gt_density).item())
        self.metrics["SoftIoU"].append(soft_iou(pred_density, gt_density).item())

        if gt_mask is not None:
            gt_mask = gt_mask.detach()
            self.metrics["DCR"].append(density_concentration_ratio(pred_density, gt_mask))
            
            pred_mask = (pred_density >= 0.5).float()
            self.metrics["Dice"].append(dice_coefficient(pred_mask, gt_mask))
            self.metrics["IoU"].append(iou(pred_mask, gt_mask))
            
            bs = pred_mask.shape[0]
            for i in range(bs):
                p_np = pred_mask[i, 0]
                g_np = gt_mask[i, 0]
                self.metrics["HD95"].append(hd95(p_np, g_np))

    def get_results(self):
        results = {}
        for k, v in self.metrics.items():
            if len(v) > 0:
                results[k] = float(np.mean(v))
            else:
                results[k] = 0.0
        return results

    def reset(self):
        self.metrics = {
            "Dice": [],
            "IoU": [],
            "HD95": [],
            "MAE": [],
            "PSNR": [],
            "SSIM": [],
            "SoftDice": [],
            "SoftIoU": [],
            "DCR": []
        }