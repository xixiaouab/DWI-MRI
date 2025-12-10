import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from glob import glob

def load_nifti(path):
    if not os.path.exists(path):
        return None
    img = nib.load(path)
    data = img.get_fdata()
    return data

def normalize_image(data):
    if data is None:
        return None
    data = data.astype(np.float32)
    min_val = np.percentile(data, 1)
    max_val = np.percentile(data, 99)
    data = np.clip(data, min_val, max_val)
    if max_val - min_val > 1e-8:
        data = (data - min_val) / (max_val - min_val)
    return data

def get_slice(data, dim, index):
    if data is None:
        return None
    if dim == 0:
        return np.rot90(data[index, :, :])
    elif dim == 1:
        return np.rot90(data[:, index, :])
    else:
        return np.rot90(data[:, :, index])

class StrokeFlowVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cmap_density = plt.cm.jet
        self.cmap_error = plt.cm.bwr
        
        colors = [(0, 0, 0, 0), (1, 0, 0, 1)]
        self.cmap_overlay = LinearSegmentedColormap.from_list("custom_overlay", colors)

    def find_best_slice(self, mask_volume):
        if mask_volume is None:
            return None
        
        dims = mask_volume.shape
        best_slices = []
        
        for d in range(3):
            sums = np.sum(mask_volume, axis=tuple(i for i in range(3) if i != d))
            best_slices.append(np.argmax(sums))
            
        return best_slices

    def plot_single_case(self, case_id, img_path, pred_dens_path, pred_flow_path, gt_dens_path=None, gt_mask_path=None):
        img_vol = load_nifti(img_path)
        pred_dens_vol = load_nifti(pred_dens_path)
        pred_flow_vol = load_nifti(pred_flow_path)
        gt_dens_vol = load_nifti(gt_dens_path) if gt_dens_path else None
        gt_mask_vol = load_nifti(gt_mask_path) if gt_mask_path else None

        if img_vol.ndim == 4:
            img_vol = img_vol[..., 0] 
        
        if pred_flow_vol.ndim == 4 and pred_flow_vol.shape[0] == 3:
            pred_flow_vol = np.transpose(pred_flow_vol, (1, 2, 3, 0))
        elif pred_flow_vol.ndim == 4 and pred_flow_vol.shape[-1] != 3:
             pred_flow_vol = np.transpose(pred_flow_vol, (1, 2, 3, 0))

        if gt_mask_vol is not None:
            best_indices = self.find_best_slice(gt_mask_vol)
        elif pred_dens_vol is not None:
            binary_pred = (pred_dens_vol > 0.5).astype(np.float32)
            if np.sum(binary_pred) > 0:
                best_indices = self.find_best_slice(binary_pred)
            else:
                best_indices = [s // 2 for s in img_vol.shape]
        else:
            best_indices = [s // 2 for s in img_vol.shape]

        self._render_views(case_id, img_vol, pred_dens_vol, pred_flow_vol, gt_dens_vol, gt_mask_vol, best_indices)

    def _render_views(self, case_id, img, pred_dens, pred_flow, gt_dens, gt_mask, indices):
        views = ['Sagittal', 'Coronal', 'Axial']
        
        fig = plt.figure(figsize=(24, 18))
        plt.style.use('dark_background')
        
        grid_rows = 3
        grid_cols = 6 
        
        gs = fig.add_gridspec(grid_rows, grid_cols)

        for i, dim_idx in enumerate([0, 1, 2]):
            slice_idx = indices[dim_idx]
            
            s_img = normalize_image(get_slice(img, dim_idx, slice_idx))
            s_pred = get_slice(pred_dens, dim_idx, slice_idx)
            
            s_flow = None
            if pred_flow is not None:
                if dim_idx == 0: 
                    s_flow = np.rot90(pred_flow[slice_idx, :, :, :]) 
                    u, v = s_flow[..., 2], s_flow[..., 1] 
                elif dim_idx == 1: 
                    s_flow = np.rot90(pred_flow[:, slice_idx, :, :])
                    u, v = s_flow[..., 2], s_flow[..., 0] 
                else: 
                    s_flow = np.rot90(pred_flow[:, :, slice_idx, :])
                    u, v = s_flow[..., 1], s_flow[..., 0] 

            s_gt_dens = get_slice(gt_dens, dim_idx, slice_idx)
            s_gt_mask = get_slice(gt_mask, dim_idx, slice_idx)

            ax_img = fig.add_subplot(gs[i, 0])
            ax_img.imshow(s_img, cmap='gray')
            ax_img.set_title(f'{views[i]} - MRI')
            ax_img.axis('off')

            ax_pred = fig.add_subplot(gs[i, 1])
            im = ax_pred.imshow(s_pred, cmap=self.cmap_density, vmin=0, vmax=1)
            ax_pred.set_title('Pred Density')
            ax_pred.axis('off')
            
            ax_flow = fig.add_subplot(gs[i, 2])
            ax_flow.imshow(s_img, cmap='gray', alpha=0.6)
            if s_flow is not None:
                h, w = s_img.shape
                step = max(h, w) // 32
                Y, X = np.mgrid[0:h:step, 0:w:step]
                
                U = u[::step, ::step]
                V = v[::step, ::step]
                
                M = np.hypot(U, V)
                Q = ax_flow.quiver(X, Y, U, V, M, cmap='autumn', scale=None, pivot='mid', width=0.005)
            ax_flow.set_title('Pred Flow Field')
            ax_flow.axis('off')

            ax_overlay = fig.add_subplot(gs[i, 3])
            ax_overlay.imshow(s_img, cmap='gray')
            ax_overlay.imshow(s_pred, cmap=self.cmap_density, alpha=0.5, vmin=0, vmax=1)
            if s_flow is not None:
                 ax_overlay.quiver(X, Y, U, V, color='white', scale=None, pivot='mid', width=0.002, alpha=0.7)
            ax_overlay.set_title('Overlay')
            ax_overlay.axis('off')

            ax_gt = fig.add_subplot(gs[i, 4])
            if s_gt_dens is not None:
                ax_gt.imshow(s_gt_dens, cmap=self.cmap_density, vmin=0, vmax=1)
                ax_gt.set_title('GT Density')
            elif s_gt_mask is not None:
                ax_gt.imshow(s_gt_mask, cmap='gray')
                ax_gt.set_title('GT Mask')
            else:
                ax_gt.text(0.5, 0.5, 'No GT', ha='center', va='center', color='white')
            ax_gt.axis('off')

            ax_diff = fig.add_subplot(gs[i, 5])
            if s_gt_dens is not None:
                diff = s_pred - s_gt_dens
                ax_diff.imshow(diff, cmap=self.cmap_error, vmin=-1, vmax=1)
                ax_diff.set_title('Error (Pred - GT)')
            else:
                 ax_diff.text(0.5, 0.5, 'N/A', ha='center', va='center', color='white')
            ax_diff.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{case_id}_comprehensive.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def process_directory(self, img_dir, pred_root, gt_root=None):
        img_files = sorted(glob(os.path.join(img_dir, "*.nii.gz")))
        
        for img_path in img_files:
            filename = os.path.basename(img_path)
            case_id = filename.split('.')[0]
            if filename.endswith('.nii.gz'):
                case_id = filename[:-7]
            elif filename.endswith('.nii'):
                case_id = filename[:-4]
            
            pred_dens_path = os.path.join(pred_root, "density", filename)
            pred_flow_path = os.path.join(pred_root, "flow", filename)
            
            gt_dens_path = None
            gt_mask_path = None
            
            if gt_root:
                gt_dens_path = os.path.join(gt_root, "density_gt", filename)
                gt_mask_path = os.path.join(gt_root, "mask_gt", filename)
                
                if not os.path.exists(gt_dens_path):
                     gt_dens_path = os.path.join(gt_root, "density_gt", case_id + ".nii.gz")
                if not os.path.exists(gt_mask_path):
                     gt_mask_path = os.path.join(gt_root, "mask_gt", case_id + ".nii.gz")

            if os.path.exists(pred_dens_path) and os.path.exists(pred_flow_path):
                self.plot_single_case(case_id, img_path, pred_dens_path, pred_flow_path, gt_dens_path, gt_mask_path)
            else:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./visualizations")
    
    args = parser.parse_args()
    
    viz = StrokeFlowVisualizer(args.output_dir)
    viz.process_directory(args.img_dir, args.pred_dir, args.gt_dir)