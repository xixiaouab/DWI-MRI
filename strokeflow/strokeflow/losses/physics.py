import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGradient3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        dz = F.pad(x, (0, 0, 0, 0, 0, 1))[:, :, 1:, :, :] - x
        dy = F.pad(x, (0, 0, 0, 1, 0, 0))[:, :, :, 1:, :] - x
        dx = F.pad(x, (0, 1, 0, 0, 0, 0))[:, :, :, :, 1:] - x
        return torch.cat([dz, dy, dx], dim=1)

class StrokeFlowLoss(nn.Module):
    def __init__(
        self,
        lambda_d=1.0,
        lambda_f=0.5,
        alpha=1.0,
        beta=0.2,
        gamma=0.1,
        lesion_threshold=0.1,
        smoothness_edge_aware=False,
        eps=1e-8
    ):
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_f = lambda_f
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lesion_threshold = lesion_threshold
        self.smoothness_edge_aware = smoothness_edge_aware
        self.eps = eps
        
        self.l1 = nn.L1Loss()
        self.grad_calculator = SpatialGradient3D()

    def _compute_alignment_loss(self, flow, density):
        grads = self.grad_calculator(density)
        target_field = -grads
        
        flow_norm = torch.norm(flow, p=2, dim=1, keepdim=True)
        target_norm = torch.norm(target_field, p=2, dim=1, keepdim=True)
        
        dot_prod = (flow * target_field).sum(dim=1, keepdim=True)
        cosine_sim = dot_prod / (flow_norm * target_norm + self.eps)
        
        align_map = 1.0 - cosine_sim
        
        mask = (density > self.lesion_threshold).float().detach()
        loss = (align_map * mask).sum() / (mask.sum() + self.eps)
        
        return loss

    def _compute_divergence_loss(self, flow):
        vz, vy, vx = flow[:, 0:1], flow[:, 1:2], flow[:, 2:3]
        
        dz_vz = F.pad(vz, (0, 0, 0, 0, 0, 1))[:, :, 1:, :, :] - vz
        dy_vy = F.pad(vy, (0, 0, 0, 1, 0, 0))[:, :, :, 1:, :] - vy
        dx_vx = F.pad(vx, (0, 1, 0, 0, 0, 0))[:, :, :, :, 1:] - vx
        
        div = dz_vz + dy_vy + dx_vx
        return div.abs().mean()

    def _compute_smoothness_loss(self, flow, image=None):
        grad_flow = self.grad_calculator(flow)
        
        if self.smoothness_edge_aware and image is not None:
            grad_img = self.grad_calculator(image).abs().mean(dim=1, keepdim=True)
            weights = torch.exp(-grad_img)
            loss = (grad_flow.abs() * weights).mean()
        else:
            loss = grad_flow.abs().mean()
            
        return loss

    def forward(self, pred_density, pred_flow, gt_density, gt_mask=None, input_image=None):
        """
        Compute combined density + flow loss.

        Args:
            pred_density: predicted scalar density field.
            pred_flow: predicted vector flow field.
            gt_density: ground-truth density.
            gt_mask: optional binary mask to focus density loss.
            input_image: optional image for edge-aware smoothness.
        """
        if gt_mask is not None:
            # Focus density regression on annotated region when provided.
            weight = (gt_mask > 0.5).float()
            diff = (pred_density - gt_density).abs() * weight
            denom = weight.sum() + self.eps
            l_density = diff.sum() / denom
        else:
            l_density = self.l1(pred_density, gt_density)
        
        l_align = self._compute_alignment_loss(pred_flow, pred_density)
        l_smooth = self._compute_smoothness_loss(pred_flow, input_image)
        l_div = self._compute_divergence_loss(pred_flow)
        
        l_flow = (self.alpha * l_align) + (self.beta * l_smooth) + (self.gamma * l_div)
        
        total_loss = (self.lambda_d * l_density) + (self.lambda_f * l_flow)
        
        return {
            "loss": total_loss,
            "L_density": l_density,
            "L_flow": l_flow,
            "L_align": l_align,
            "L_smooth": l_smooth,
            "L_div": l_div
        }